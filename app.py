from typing import TypedDict, List
from langgraph.graph import StateGraph,END
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from ingest_pdf import function

import asyncio
import os
from dotenv import load_dotenv

# ===== MCP =====
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# ---------------- ENV ----------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")

llm = ChatGroq(model="qwen/qwen3-32b")

MCP_URL = "http://localhost:3001/mcp"

# ---------------- MCP SAFE CALL ----------------
async def call_mcp(tool: str, args: dict):
    try:
        async with sse_client(url=MCP_URL) as stream:
            await stream.send(tool, args)
            async for msg in stream:
                return msg if isinstance(msg, list) else [str(msg)]
                    
    except Exception:
        return []

# ---------------- RAG ----------------
# db = Chroma(
#     persist_directory="chroma_db",
#     embedding_function=HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
# )
db=function()


retriever = db.as_retriever(search_kwargs={"k": 3})

# ---------------- STATE ----------------
class State(TypedDict, total=False):
    question: str
    context: str
    sources: List[str]
    next_tool: str
    final: str

# ---------------- AGENT ----------------
def agent_node(state: State):
    prompt = f"""
You are a routing agent.

Choose the BEST source:

Rules:
- Use "pdf" ONLY if the question is strictly about:
  GenAI, Diffusion model, embeddings, vision models,GANs, VAEs, LLMs,Tokenization,prompt engineering
- Use "duckduckgo" for:
  company definitions, general knowledge, history, education
- Use "tavily" for:
  latest updates, news, current events, weather

Return ONLY one word:
pdf
duckduckgo
tavily

Question:
{state['question']}
"""
    decision = llm.invoke(prompt).content.strip().lower()

    if decision not in {"pdf", "duckduckgo", "tavily"}:
        decision = "duckduckgo"

    return {"next_tool": decision}



# ---------------- PDF ----------------
def pdf_node(state: State):
    docs = retriever.invoke(state["question"])

    if not docs:
        return {"pdf_confidence": "low"}

    context = docs[0].page_content[:800]

    judge_prompt = f"""
Question:
{state['question']}

Context:
{context}

Is this context useful to answer the question?
Reply ONLY yes or no.
"""

    verdict = llm.invoke(judge_prompt).content.lower().strip()

    if "yes" in verdict:
        return {
            "pdf_confidence": "high",
            "context": "\n\n".join(d.page_content for d in docs),
            "sources": ["PDF Document"]
        }

    return {"pdf_confidence": "low"}


def after_pdf_decision(state: State):
    if state.get("pdf_confidence") == "high":
        return "final"
    return "duckduckgo"


# ---------------- DUCK ----------------
async def duck_node(state: State):
    res = await call_mcp("duckduckgo_search", {"query": state["question"]})

    return {
        "context": state.get("context", "") + "\n" + "\n".join(res),
        "sources": state.get("sources", []) + ["DuckDuckGo"]
    }

# ---------------- TAVILY ----------------
async def tavily_node(state: State):
    res = await call_mcp("tavily_search", {"query": state["question"]})

    return {
        "context": state.get("context", "") + "\n" + "\n".join(res),
        "sources": state.get("sources", []) + ["Tavily"]
    }

# ---------------- FINAL ----------------
def final_node(state: State):
    prompt = f"""
You are a helpful assistant.

STRICT RULES:
- DO NOT show your thinking
- DO NOT include <think> tags
- DO NOT explain reasoning
- ONLY return the final answer

Use ONLY the context below.

Context:
{state.get("context", "")}

Question:
{state["question"]}

Answer:
"""

    answer = llm.invoke(prompt).content

    return {
        "final": f"""
Answer:
{answer}

Sources:
{', '.join(state.get('sources', []))}
"""
    }



# ---------------- GRAPH ----------------
graph = StateGraph(State)

graph.add_node("pdf", pdf_node)
graph.add_node("agent", agent_node)
graph.add_node("duck", duck_node)
graph.add_node("tavily", tavily_node)
graph.add_node("final", final_node)

# ALWAYS START WITH RETRIEVAL
graph.set_entry_point("pdf")

graph.add_conditional_edges(
    "pdf",
    after_pdf_decision,
    {
        "final": "final",
        "duckduckgo": "agent"
    }
)

graph.add_conditional_edges(
    "agent",
    lambda s: s["next_tool"],
    {
        "duckduckgo": "duck",
        "tavily": "tavily"
    }
)

graph.add_edge("duck", "final")
graph.add_edge("tavily", "final")

app = graph.compile()



# ---------------- RUN ----------------
async def main():
    while True:
        q = input("\nAsk a question (or exit): ")
        if q.lower() == "exit":
            break

        result = await app.ainvoke({"question": q})
        print(result["final"])

if __name__ == "__main__":
    asyncio.run(main())
