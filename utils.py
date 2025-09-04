from typing import Optional
from openai import OpenAI
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel
import os


def get_model():
    kwargs = {
        "base_url": "http://localhost:1234/v1",
        # API key is not required for the local server, but a placeholder is needed by the SDK
        "api_key": "lm-studio",
    }
    return OpenAI(**kwargs)
    # return Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))


def invoke_model(
    # model: OpenAI,
    model: Cerebras,
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
    # model: OpenAI,
    model: Cerebras,
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
