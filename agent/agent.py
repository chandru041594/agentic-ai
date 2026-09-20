from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_community.llms import HuggingFaceHub

# Define schema for the agent state
class AgentState(TypedDict):
    input: str
    context: str
    output: str

# Dummy RAG node (replace with real retriever later)
def rag_node(state: AgentState):
    return {"context": "retrieved docs"}

# LLM node
def llm_node(state: AgentState):
    llm = HuggingFaceHub(repo_id="google/flan-t5-small")
    response = llm(state["context"] + state["input"])
    return {"output": response}

# Build graph with schema
graph = StateGraph(AgentState)
graph.add_node("RAG", rag_node)
graph.add_node("LLM", llm_node)
graph.add_edge("RAG", "LLM")
graph.set_entry_point("RAG")
