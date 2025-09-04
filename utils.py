from openai import OpenAI


def get_model():
    kwargs = {
        "base_url": "http://localhost:1234/v1",
        # API key is not required for the local server, but a placeholder is needed by the SDK
        "api_key": "lm-studio",
    }
    return OpenAI(**kwargs)


def invoke_model(model: OpenAI, prompt: str):
    return model.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # model="gpt-4o-mini",
        model="google/gemma-3n-e4b",
    )
