from dotenv import load_dotenv
import logging
from tools.lookup_sales import lookup_sales_data


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

def main():
    lookup_sales_data("Show me all the sales for store 1320 on November 1st, 2021")


if __name__ == "__main__":
    main()
