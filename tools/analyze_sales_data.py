from openai import OpenAI

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis."""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    print("Performing data analysis")
    response = OpenAI().chat.completions.create(messages=[
        {
            "role": "user",
            "content": formatted_prompt,
        }
    ], model="gpt-4o-mini")
    print("Performed data analysis with model")

    return response.choices[0].message.content or "No analysis available"
