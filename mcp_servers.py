import os
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from ddgs import DDGS
from tavily import TavilyClient

# ---------------- ENV ----------------
load_dotenv()

# ---------------- MCP SERVER ----------------
mcp = FastMCP("web-tools")

# ---------------- TAVILY ----------------
tavily = TavilyClient(api_key=os.getenv("TAVILY_API"))

# ---------------- DUCKDUCKGO TOOL ----------------
@mcp.tool()
def duckduckgo_search(query: str) -> list[str]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(f"{r['title']} - {r['href']}")
    return results

# ---------------- TAVILY TOOL ----------------
@mcp.tool()
def tavily_search(query: str) -> list[str]:
    res = tavily.search(query=query, max_results=3)
    return [f"{r['title']} - {r['url']}" for r in res["results"]]

# ---------------- RUN ----------------
if __name__ == "__main__":
    mcp.run()
