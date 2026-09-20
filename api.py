
from fastapi import FastAPI
from agent.agent import graph   # make sure agent/agent.py exists with __init__.py

app = FastAPI()

@app.get("/")   # root endpoint
def read_root():
    return {"message": "Agentic AI API is running"}

@app.get("/health")   # quick health check
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(query: str):
    result = graph.run({"input": query})
    return {"response": result["output"]}

