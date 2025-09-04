from typing import Optional
from openai import OpenAI
from pydantic import BaseModel


def get_model():
    kwargs = {
        "base_url": "http://localhost:1234/v1",
        # API key is not required for the local server, but a placeholder is needed by the SDK
        "api_key": "lm-studio",
    }
    return OpenAI(**kwargs)


def invoke_model(
    model: OpenAI,
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
        tools=tools,
    )


def invoke_model_with_structured_output(
    model: OpenAI,
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
        tools=tools,
        response_format=output_format,
    )
