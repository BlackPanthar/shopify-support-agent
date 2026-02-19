# =============================================================================
# SHOPIFY CUSTOMER SUPPORT AGENT
# =============================================================================
#
# ARCHITECTURE OVERVIEW (read this first!):
#
#   User types message
#        ↓
#   Streamlit UI  ──→  AgentExecutor  ──→  LLM (GPT-4o-mini)
#                            ↓                    ↓
#                       decides which        generates reply
#                       tool(s) to call
#                            ↓
#                   Tool: get_order_status()   ← queries mock data (or real Shopify later)
#                   Tool: process_return_or_refund()
#
#   Memory is stored in Streamlit's session_state so it survives page reruns.
#
# =============================================================================

import os
from dotenv import load_dotenv

# LangChain imports — each has a specific role:
from langchain_openai import ChatOpenAI           # Wrapper around OpenAI's API
# NOTE: In LangChain 1.x, the classic agent components moved to langchain_classic
from langchain_classic.agents import (
    create_tool_calling_agent,                    # Builds the "brain" that decides which tools to call
    AgentExecutor,                                # Runs the agent in a loop until it has a final answer
)
from langchain_core.tools import tool             # Decorator that turns a Python function into an agent tool
from langchain_core.prompts import (
    ChatPromptTemplate,                           # Template that structures messages sent to the LLM
    MessagesPlaceholder,                          # Placeholder that gets filled with dynamic message history
)
from langchain_core.messages import HumanMessage, AIMessage  # Typed message objects

import streamlit as st                            # Web UI framework — turns this script into a web app

# Load .env file so os.getenv() can read our secret keys
load_dotenv()


# =============================================================================
# STEP 1: SET UP THE LLM VIA OPENROUTER
# =============================================================================
#
# OpenRouter is a single API that routes to 100s of models:
#   GPT-4o, Claude 3.5, Gemini 2.0, Llama 3, Mistral, etc.
# One API key, one endpoint, swap models by changing one string.
#
# WHY OpenRouter instead of OpenAI directly?
#   - Free models available (no card needed)
#   - Cheaper paid models than going direct
#   - Easy to compare models — just change the model name
#   - Same API format as OpenAI so langchain_openai works unchanged
#
# HOW IT WORKS:
#   We use langchain_openai's ChatOpenAI but point it at OpenRouter's URL.
#   OpenRouter then forwards the request to the actual model provider.
#   Your app doesn't need to know which company runs the model.
#
# FREE MODELS (no cost, no card):
#   "google/gemini-2.0-flash-exp:free"
#   "meta-llama/llama-3.1-8b-instruct:free"
#   "mistralai/mistral-7b-instruct:free"
#   "qwen/qwen3-235b-a22b:free"
#   (see all at openrouter.ai/models — filter by "Free")
#
# PAID MODELS (very cheap, great quality):
#   "openai/gpt-4o-mini"         ← $0.15/1M tokens
#   "anthropic/claude-3.5-haiku" ← fast, very capable
#   "google/gemini-2.0-flash"    ← great for tool calling
#
# temperature=0 → deterministic (same input = same output).
#   Good for support agents. Raise to 0.7 for creative tasks.
# =============================================================================

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",   # Cheap paid model — $0.15/1M tokens, excellent tool calling
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:8501",   # Tells OpenRouter where the request came from
        "X-Title": "Shopify Support Agent",        # Shows in your OpenRouter dashboard
    },
)


# =============================================================================
# STEP 2: DEFINE MOCK DATA
# =============================================================================
#
# This simulates a real database/Shopify store. We'll replace these
# with real Shopify API calls in Phase 3.
#
# WHY mock first? You can build and test the entire agent logic without
# needing API keys or a real store. This is called "mocking" — standard
# practice in software development.
# =============================================================================

MOCK_ORDERS = {
    "1001": {
        "status": "open",
        "fulfillment_status": "fulfilled",
        "tracking_number": "1Z9999999999",
        "carrier": "UPS",
        "total_price": "89.99",
        "items": ["Blue T-Shirt (M)", "Black Hoodie (L)"],
        "customer_email": "jane@example.com",
    },
    "1002": {
        "status": "open",
        "fulfillment_status": "unfulfilled",
        "tracking_number": None,
        "carrier": None,
        "total_price": "45.50",
        "items": ["Red Coffee Mug"],
        "customer_email": "bob@example.com",
    },
    "1003": {
        "status": "closed",
        "fulfillment_status": "fulfilled",
        "tracking_number": "9400111899223450487758",
        "carrier": "USPS",
        "total_price": "129.00",
        "items": ["Wireless Earbuds", "Phone Case"],
        "customer_email": "alice@example.com",
    },
}


# =============================================================================
# STEP 3: DEFINE TOOLS
# =============================================================================
#
# Tools are the "hands" of the agent — specific actions it can take.
# The @tool decorator:
#   1. Registers the function so LangChain knows it exists
#   2. Uses the docstring as the description the LLM reads to decide WHEN
#      to call this tool. Write clear docstrings — the LLM uses them!
#   3. Uses type hints (order_id: str) to tell the LLM what arguments to pass
#
# IMPORTANT: The LLM doesn't call tools directly. It outputs a JSON blob like:
#   {"tool": "get_order_status", "args": {"order_id": "1001"}}
# Then AgentExecutor actually runs the Python function and feeds the result
# back to the LLM for the final response.
# =============================================================================

@tool
def get_order_status(order_id: str) -> str:
    """
    Fetch the current status, fulfillment status, and tracking number for a
    given order ID. Use this when a customer asks where their order is,
    what the status is, or for a tracking number.
    """
    # Strip whitespace and leading # in case user types "#1001"
    order_id = order_id.strip().lstrip("#")

    # Validate: order IDs should be numeric
    if not order_id.isdigit():
        return f"Invalid order ID '{order_id}'. Order IDs should be numbers only (e.g. 1001)."

    order = MOCK_ORDERS.get(order_id)
    if not order:
        return f"Order #{order_id} was not found in our system. Please double-check the order number."

    items_list = ", ".join(order["items"])
    fulfillment = order["fulfillment_status"].capitalize()

    if order["tracking_number"]:
        tracking_info = f"Tracking: {order['tracking_number']} via {order['carrier']}"
    else:
        tracking_info = "Tracking: Not yet shipped — no tracking number available yet"

    return (
        f"Order #{order_id} Details:\n"
        f"  • Status: {order['status'].capitalize()}\n"
        f"  • Fulfillment: {fulfillment}\n"
        f"  • {tracking_info}\n"
        f"  • Items: {items_list}\n"
        f"  • Total: ${order['total_price']}"
    )


@tool
def process_return_or_refund(order_id: str, reason: str = "Customer request") -> str:
    """
    Initiate a return or refund for a given order ID. Use this when a customer
    wants to return an item, get a refund, or reports a problem with their order.
    Always confirm the order exists before processing.
    """
    order_id = order_id.strip().lstrip("#")

    if not order_id.isdigit():
        return f"Invalid order ID '{order_id}'. Order IDs should be numbers only."

    if order_id not in MOCK_ORDERS:
        return f"Cannot process: Order #{order_id} not found."

    order = MOCK_ORDERS[order_id]

    # Business logic: can't refund an order that hasn't shipped yet
    if order["fulfillment_status"] == "unfulfilled":
        return (
            f"Order #{order_id} hasn't shipped yet. We've cancelled the order instead "
            f"and a full refund of ${order['total_price']} will appear in 3-5 business days."
        )

    # In a real implementation, this is where you'd call:
    # shopify.Order.find(order_id).refund(...)
    return (
        f"Return/refund initiated for Order #{order_id}.\n"
        f"  • Reason: {reason}\n"
        f"  • Refund amount: ${order['total_price']}\n"
        f"  • Timeline: 3-5 business days back to original payment method\n"
        f"  • A confirmation email has been sent to {order['customer_email']}"
    )


# Bundle tools into a list — the agent gets access to all of these
tools = [get_order_status, process_return_or_refund]


# =============================================================================
# STEP 4: CREATE THE PROMPT TEMPLATE
# =============================================================================
#
# The prompt is the instruction manual for the LLM. It tells the AI:
#   - Who it is ("You are a friendly support agent...")
#   - What it can/can't do
#   - How to behave
#
# ChatPromptTemplate.from_messages() takes a list of (role, content) tuples:
#   - "system": Background instructions (the LLM "reads" this before everything)
#   - "human": The user's message
#   - MessagesPlaceholder: A slot that gets filled with dynamic content at runtime
#
# {chat_history} → filled with previous messages (our memory)
# {input}        → filled with the current user message
# {agent_scratchpad} → filled with the agent's internal reasoning steps
# =============================================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly and professional customer support agent for an online Shopify store.

Your responsibilities:
- Help customers track their orders
- Process returns and refunds
- Provide order status updates

Guidelines:
- Always ask for an order ID if the customer hasn't provided one
- Be empathetic and apologetic when customers have problems
- Keep responses concise and clear
- If a query is unrelated to orders/returns/tracking, politely redirect:
  "I specialize in order support — tracking, returns, and refunds. For other questions, please contact us at support@store.com"

You have access to tools that can look up real order data. Always use them rather than guessing."""),

    # This slot holds all previous messages in the conversation
    MessagesPlaceholder(variable_name="chat_history"),

    # This is where the user's current message goes
    ("human", "{input}"),

    # This is where the agent writes its internal "thinking" (tool calls, etc.)
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# =============================================================================
# STEP 5: ASSEMBLE THE AGENT
# =============================================================================
#
# create_tool_calling_agent() combines:
#   - The LLM (the brain)
#   - The tools (the hands)
#   - The prompt (the instructions)
# → Returns an agent "runnable" (a LangChain object you can invoke)
#
# AgentExecutor wraps the agent and runs the "Thought → Action → Observation" loop:
#   1. Agent thinks about what to do
#   2. Calls a tool (Action)
#   3. Gets tool result (Observation)
#   4. Repeats until it has enough info to give a final answer
#
# verbose=True prints this loop to your terminal — great for debugging!
# =============================================================================

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # Print reasoning steps in terminal (set False in production)
    handle_parsing_errors=True,  # Gracefully handle malformed LLM outputs
)


# =============================================================================
# STEP 6: STREAMLIT UI
# =============================================================================
#
# Streamlit works differently from regular Python scripts:
#   - The ENTIRE script re-runs from top to bottom on EVERY user interaction
#   - st.session_state is a dictionary that PERSISTS between reruns
#   - This is why we store chat history in session_state, not a regular variable
#
# Flow:
#   User types message → Streamlit reruns script → we read session_state
#   → invoke agent → store new messages → Streamlit renders updated chat
# =============================================================================

st.set_page_config(
    page_title="Shopify Support Agent",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ E-Commerce Support Agent")
st.caption("Powered by GPT-4o-mini + LangChain | Ask about orders 1001, 1002, or 1003")

# Sidebar with demo info
with st.sidebar:
    st.header("Demo Orders")
    st.markdown("""
    Try these test scenarios:

    **Order 1001** — Fulfilled (shipped)
    - 2 items, has tracking number

    **Order 1002** — Unfulfilled (not shipped)
    - 1 item, no tracking yet

    **Order 1003** — Closed/delivered
    - 2 items, USPS tracking

    ---
    **Sample queries:**
    - "Where is order 1001?"
    - "I want to return order #1003"
    - "Can you track my order 1002?"
    - "Refund order 1001, wrong size"
    """)

    # Button to clear conversation history
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Initialize session state on first load
# WHY: Session state persists between Streamlit reruns (page interactions)
# Without this, memory would reset on every message
if "messages" not in st.session_state:
    st.session_state.messages = []        # Stores messages for display (role + content)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # Stores LangChain message objects for the agent

# Display all previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box at the bottom of the page
# := is Python's "walrus operator" — assigns and checks in one line
if user_input := st.chat_input("Ask about your order (e.g. 'Where is order 1001?')"):

    # 1. Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Run the agent and show a spinner while it's thinking
    with st.chat_message("assistant"):
        with st.spinner("Looking that up..."):
            try:
                # Invoke the agent, passing:
                # - "input": the user's current message
                # - "chat_history": all previous messages (gives the agent memory)
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                })
                output = response["output"]

                # Update chat history with this exchange (for future turns)
                # HumanMessage and AIMessage are typed wrappers LangChain understands
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=output),
                ])

            except Exception as e:
                output = f"Sorry, something went wrong: {str(e)}"

            st.markdown(output)

    # 3. Save assistant response to display history
    st.session_state.messages.append({"role": "assistant", "content": output})
