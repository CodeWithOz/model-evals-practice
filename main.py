from dotenv import load_dotenv
import logging
from tools import AgentRouter


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()


def main():
    agent_res = AgentRouter().run_agent(
        "Show me the code for graph of sales by store in Nov 2021, and tell me what trends you see."
    )
    print(f"\n\nAgent final response: {agent_res}")


if __name__ == "__main__":
    main()
