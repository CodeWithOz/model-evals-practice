from dotenv import load_dotenv

load_dotenv()

import logging

from opentelemetry.trace import StatusCode
from tools import AgentRouter
from utils import get_tracer


tracer = get_tracer(__name__)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    with tracer.start_as_current_span(
        "agent_run", openinference_span_kind="agent"
    ) as span:
        user_query = "Show me the code for graph of sales by store in Nov 2021, and tell me what trends you see."
        span.set_input(value=user_query)
        agent_res = AgentRouter().run_agent(user_query)
        span.set_output(value=agent_res)
        span.set_status(StatusCode.OK)
        print(f"\n\nAgent final response: {agent_res}")


if __name__ == "__main__":
    main()
