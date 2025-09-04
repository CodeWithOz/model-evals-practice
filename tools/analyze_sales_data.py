from openai import OpenAI

from utils import get_model, invoke_model

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis."""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    print("Performing data analysis")
    response = invoke_model(get_model(), formatted_prompt)
    print("Performed data analysis with model")

    analysis = response.choices[0].message.content or "No analysis available"

    print(f"Analysis:\n\n{analysis}")

    return analysis
