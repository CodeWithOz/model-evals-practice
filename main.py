from dotenv import load_dotenv
import logging
from tools.analyze_sales_data import analyze_sales_data
from tools.lookup_sales import lookup_sales_data
from tools.visualize_data import (
    generate_visualization,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()


def main():
    sales_data = lookup_sales_data(
        "Show me all the sales for store 1320 on November 1st, 2021"
    )
    analyze_sales_data("what trends do you see in this data", sales_data)

    chart_code = generate_visualization(
        sales_data,
        "A bar chart of sales by product SKU. Put the product SKU on the x-axis and the sales on the y-axis.",
    )

    print("Executing generated code")
    try:
        exec(chart_code)
        print("Executed generated code successfully")
    except Exception as e:
        print(f"Error executing generated code: {str(e)}")


if __name__ == "__main__":
    main()
