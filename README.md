# AI Task Agent — Tools & Automation

A tool-using AI agent built with Python, OpenAI function calling, Streamlit and SQLite.

## What it can do
- Calculate basic arithmetic using a restricted calculator tool.
- Add tasks to a local SQLite database when explicitly requested.
- List saved tasks.
- Search a small built-in knowledge base.
- Decide when to call tools, process tool results and produce a final response.
- Keep chat history during the Streamlit session.

## Agent workflow
User request → Understand task → Decide whether a tool is needed → Call tool → Process result → Respond.

## Tools
1. `calculate(expression)`: basic arithmetic only; uses a restricted AST evaluator.
2. `add_task(task)`: stores a task in SQLite.
3. `list_tasks()`: reads saved tasks from SQLite.
4. `search_knowledge(query)`: searches sample learning notes stored in SQLite.

## Run locally
1. Install Python 3.10+.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`.
4. Add your own API key to `.env`. Never commit `.env`.
5. Run: `streamlit run app.py`

The application uses the OpenAI API for model responses and tool selection. API usage may incur charges. Set `OPENAI_MODEL` to a model available to your account.

## Example prompts
- `Calculate (125 * 4) + 30.`
- `Add a task: revise Python function calling.`
- `Show my saved tasks.`
- `Search the notes for embeddings.`

## Safety and limitations
- The calculator accepts basic arithmetic only; it does not execute Python code.
- The agent has a bounded tool-call loop to prevent runaway tool use.
- Task additions write to the local SQLite database. The app only adds tasks when the user explicitly asks.
- This is an educational prototype, not a production agent. Add authentication, access controls, monitoring and robust persistence before deployment.
- Do not place secrets in source code or public repositories.
