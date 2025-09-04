from pydantic import BaseModel, Field

from utils import get_model, invoke_model_with_structured_output

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
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

    chart_config: VisualizationConfig = (
        invoke_model_with_structured_output(get_model(), formatted_prompt, VisualizationConfig)
    ).choices[0].message.parsed

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
