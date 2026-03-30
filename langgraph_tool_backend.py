from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
 
load_dotenv()  # loads GOOGLE_API_KEY from .env automatically
 
# ─────────────────────────────────────────
# 1.  LLM
# ─────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)
 
# ─────────────────────────────────────────
# 2.  System prompt
# ─────────────────────────────────────────
SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful, concise assistant. "
        "Format code blocks with proper markdown fencing."
    )
)
 
# ─────────────────────────────────────────
# 3.  Graph node
# ─────────────────────────────────────────
def call_model(state: MessagesState):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
 
# ─────────────────────────────────────────
# 4.  Build graph
# ─────────────────────────────────────────
builder = StateGraph(MessagesState)
builder.add_node("llm", call_model)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
 
memory = MemorySaver()
 
# ─────────────────────────────────────────
# 5.  Compile  ← exported as 'chatbot' for the frontend
# ─────────────────────────────────────────
chatbot = builder.compile(checkpointer=memory)