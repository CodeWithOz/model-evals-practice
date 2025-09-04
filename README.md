# Model Evaluations Practice

This project demonstrates practical techniques for evaluating AI agents using real-world sales data analysis. It implements an agent-based system that can analyze sales data, generate visualizations, and answer questions about business metrics.

## Features

- **Agent Router**: Intelligent routing between different AI models (Cerebras, OpenAI, local LLMs)
- **Sales Analysis Tools**:
  - Sales data lookup and querying
  - Data visualization generation
  - Trend analysis and insights

## Setup

1. Copy `.env.sample` to `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   CEREBRAS_API_KEY=your_cerebras_key_here
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Run the main agent:
   ```bash
   uv run main.py
   ```

## Data

The project uses a sales dataset (`Store_Sales_Price_Elasticity_Promotions_Data.parquet`) containing store-level sales data with promotional information for analysis and evaluation.

## Course Context

This project is part of the **Evaluating AI Agents** short course available at [DeepLearning.AI](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/). The course covers best practices for building and evaluating AI agent systems, including prompt engineering, tool use, and performance metrics.