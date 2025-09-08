import json
from opentelemetry.trace import StatusCode
from tools.analyze_sales_data import analyze_sales_data
from tools.lookup_sales import lookup_sales_data
from tools.visualize_data import generate_visualization
from utils import get_model, invoke_model, get_tracer


tracer = get_tracer(__name__)


class AgentRouter:
    SYSTEM_PROMPT = """
    You are a helpful assistant that can answer questions about
    the Store Sales Price Elasticity Promotions dataset.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_sales_data",
                "description": "Look up data from Store Sales Price Elasticity Promotions dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The unchanged prompt that the user provided.",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_sales_data",
                "description": "Analyze sales data to extract insights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The unchanged prompt that the user provided.",
                        },
                        "data": {
                            "type": "string",
                            "description": "The output of the lookup_sales_data tool.",
                        },
                    },
                    "required": ["prompt", "data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_visualization",
                "description": "Generate code to visualize data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "The output of the lookup_sales_data tool",
                        },
                        "visualization_goal": {
                            "type": "string",
                            "description": "The goal of the visualization",
                        },
                    },
                    "required": ["data", "visualization_goal"],
                },
            },
        },
    ]

    tool_implementations = {
        "lookup_sales_data": lookup_sales_data,
        "analyze_sales_data": analyze_sales_data,
        "generate_visualization": generate_visualization,
    }

    def __init__(self):
        pass

    @tracer.chain()
    def handle_tool_calls(self, tool_calls: list[dict], messages: list):
        tool_results = {}
        print(f"Invoking {len(tool_calls)} tool calls")
        for tool_call in tool_calls:
            function = self.tool_implementations[tool_call.function.name]
            function_args = json.loads(tool_call.function.arguments)
            tool_res = function(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "content": tool_res,
                    "tool_call_id": tool_call.id,
                }
            )
            tool_results.setdefault(tool_call.id, tool_res)

        print(f"Invoked {len(tool_calls)} tool calls")

        return messages, tool_results

    def run_agent(self, messages: list | str):
        print("Running agent with messages:", messages)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if not any(
            isinstance(me, dict) and me.get("role") == "system" for me in messages
        ):
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages

        while True:
            print("Making router call to model provider")
            with tracer.start_as_current_span(
                "router_call", openinference_span_kind="chain"
            ) as span:
                span.set_input(value=messages)
                response = invoke_model(
                    get_model(), messages=messages, tools=self.tools
                )
                message = response.choices[0].message
                span.set_status(StatusCode.OK)
                message_content = (
                    message.content or "Router returned empty message content"
                )
                print(f"Received router response: {message_content!r}")
                messages.append(message)

                # handle tool calls if necessary
                if message.tool_calls:
                    print(
                        f"Router response contained {len(message.tool_calls)} tool calls"
                    )
                    _, tool_results = self.handle_tool_calls(
                        message.tool_calls, messages
                    )
                    span_output = {
                        tool_call.id: {
                            **json.loads(tool_call.to_json()),
                            "tool_call_res": tool_results.get(tool_call.id, None),
                        }
                        for tool_call in message.tool_calls
                    }
                    span.set_output(value=span_output)
                else:
                    # no tool calls, return the model's final response
                    print("Router created no tool calls, returning final response")
                    span.set_output(value=message_content)
                    return message_content
