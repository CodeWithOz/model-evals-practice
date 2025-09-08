from typing import Optional
from openai import OpenAI
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel
import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from dotenv import load_dotenv


load_dotenv()


PROJECT_NAME = "model-evals-practice"


arize_tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") + "/v1/traces",
)

OpenAIInstrumentor().instrument(tracer_provider=arize_tracer_provider)


def get_tracer(file_name: str):
    return arize_tracer_provider.get_tracer(file_name)


def get_model():
    kwargs = {
        "base_url": "http://localhost:1234/v1",
        # API key is not required for the local server, but a placeholder is needed by the SDK
        "api_key": "lm-studio",
    }
    return OpenAI(**kwargs)
    # return Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))


def invoke_model(
    model: OpenAI,
    # model: Cerebras,
    prompt: Optional[str] = None,
    messages: Optional[list] = None,
    tools: Optional[list] = None,
):
    if messages is None:
        messages = []

    if tools is None:
        tools = []

    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})

    return model.chat.completions.create(
        messages=messages,
        # model="gpt-4o-mini",
        model="google/gemma-3n-e4b",
        # model="gpt-oss-120b",
        tools=tools,
    )


def invoke_model_with_structured_output(
    model: OpenAI,
    # model: Cerebras,
    output_format: BaseModel,
    prompt: Optional[str] = None,
    messages: Optional[list] = None,
    tools: Optional[list] = None,
):
    if messages is None:
        messages = []

    if tools is None:
        tools = []

    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})

    return model.chat.completions.parse(
        messages=messages,
        # model="gpt-4o-mini",
        model="google/gemma-3n-e4b",
        # model="gpt-oss-120b",
        tools=tools,
        response_format=output_format,
    )
