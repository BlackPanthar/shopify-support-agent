# Shopify Customer Support Agent

An AI-powered customer support agent built with LangChain and GPT-4o-mini. Handles order tracking, returns, and refunds through natural language conversation.

## Demo

> "Where is order 1001?" → Agent fetches order status and tracking number
> "Refund order 1002, wrong size" → Agent processes return and confirms
> "What's the weather?" → Agent politely deflects off-topic queries

## Features

- **Natural language** — customers ask in plain English, no forms
- **Tool calling** — agent decides which action to take and executes it
- **Conversation memory** — handles follow-ups ("refund it" after asking about an order)
- **Shopify-ready architecture** — mock data swaps to real Shopify API with one function change
- **Clean web UI** — built with Streamlit, deployable in minutes

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini via OpenRouter |
| Agent Framework | LangChain (tool calling agent) |
| UI | Streamlit |
| API | Shopify Admin API (mocked for demo) |

## Setup

**1. Clone the repo**
```bash
git clone <your-repo-url>
cd shopify-support-agent
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your API key**
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

**4. Run**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Get from [openrouter.ai/keys](https://openrouter.ai/keys) |
| `SHOPIFY_STORE_URL` | Your store URL (for real API integration) |
| `SHOPIFY_ACCESS_TOKEN` | From Shopify custom app admin |

## Project Structure

```
shopify-support-agent/
├── app.py              # Main agent logic + Streamlit UI
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

## How It Works

```
User message
     ↓
Streamlit UI → AgentExecutor → LLM (GPT-4o-mini)
                    ↓                ↓
              picks a tool     writes reply
                    ↓
         get_order_status()
         process_return_or_refund()
```

The agent uses **tool calling** — the LLM outputs structured JSON specifying which function to call and with what arguments. LangChain executes the function and feeds the result back to the LLM for the final response.

## Extending to Real Shopify

Replace the mock functions in `app.py` with real API calls:

```python
import shopify

@tool
def get_order_status(order_id: str) -> str:
    order = shopify.Order.find(order_id)
    return f"Order #{order_id}: {order.fulfillment_status}, tracking: {order.tracking_number}"
```
