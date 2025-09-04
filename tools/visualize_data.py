from pydantic import BaseModel, Field

from utils import get_model, invoke_model, invoke_model_with_structured_output

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""


CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.

Only return the code, no other text.

config: {config}
"""


class VisualizationConfig(BaseModel):
    """Defines the response format of step 1"""

    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")


def extract_chart_config(data: str, visualization_goal: str) -> dict:
    """
    Generate chart visualization configuration

    Args:
        data: String containing the data to visualize
        visualization_goal: Description of what the visualization should show

    Returns:
        Dictionary containing line chart config
    """
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
        data=data, visualization_goal=visualization_goal
    )

    print("Generating chart config with model")
    chart_config: VisualizationConfig = (
        (
            invoke_model_with_structured_output(
                get_model(), VisualizationConfig, prompt=formatted_prompt
            )
        )
        .choices[0]
        .message.parsed
    )

    print(f"Generated chart config:\n\n{chart_config.model_dump_json()}")

    try:
        return {
            "chart_type": chart_config.chart_type,
            "x_axis": chart_config.x_axis,
            "y_axis": chart_config.y_axis,
            "title": chart_config.title,
            "data": data,
        }
    except Exception as e:
        print(f"Failed to generate chart config: {e}")
        return {
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data,
        }


def create_chart(config: dict):
    """Create a chart based on the configuration"""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)

    print("Generating chart code with model")
    chart_code = (
        invoke_model(get_model(), formatted_prompt).choices[0].message.content
        or "No chart code available"
    )
    chart_code = chart_code.replace("```python", "").replace("```", "").strip()

    print(f"Generated chart code:\n\n{chart_code}")

    return chart_code


def generate_visualization(data: str, visualization_goal: str) -> str:
    chart_config = extract_chart_config(data, visualization_goal)
    chart_code = create_chart(chart_config)

    return chart_code
