from dotenv import load_dotenv
import tqdm
import nest_asyncio
import warnings
import logging
from opentelemetry.trace import StatusCode
from tools import AgentRouter
from utils import get_tracer


load_dotenv()


tracer = get_tracer(__name__)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore")
nest_asyncio.apply()


def main():
    agent_questions = [
        "What was the most popular product SKU?",
        "What was the total revenue across all stores?",
        "Which store had the highest sales volume?",
        "Create a bar chart showing total sales by store",
        "What percentage of items were sold on promotion?",
        "What was the average transaction value?",
    ]
    for question in tqdm.tqdm(agent_questions, desc="Processing questions"):
        with tracer.start_as_current_span(
            "agent_run", openinference_span_kind="agent"
        ) as span:
            span.set_input(value=question)
            agent_res = AgentRouter().run_agent(question)
            span.set_output(value=agent_res)
            span.set_status(StatusCode.OK)
            print(f"\n\nAgent final response: {agent_res}")


if __name__ == "__main__":
    main()
