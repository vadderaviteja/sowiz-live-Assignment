# PDF + Web Hybrid RAG System using LangGraph, MCP & Groq

This project implements a hybrid Retrieval-Augmented Generation (RAG) system that intelligently answers user questions by prioritizing PDF documents first and falling back to web search tools (DuckDuckGo, Tavily) when required.

It uses LangGraph for orchestration, ChromaDB for vector storage, MCP (Model Context Protocol) for tool calling, and Groq LLMs for fast inference.

# Features

📘 PDF-first RAG using embeddings + vector database

🧠 Intelligent routing agent (PDF vs DuckDuckGo vs Tavily)

🔁 Confidence-based fallback from PDF → Web search

🌐 Web tools via MCP

DuckDuckGo (general knowledge)

Tavily (latest updates, weather, news)

🧩 LangGraph-based workflow

🚫 No chain-of-thought leakage (<think> removed)

⚡ Fast inference using Groq (Qwen 3 32B)

# 🏗️ Architecture Overview

User Question
      |
      v
 PDF Retrieval (Chroma)
      |
      |-- if confident --> Final Answer
      |
      v
 Routing Agent (LLM)
      |
      +--> DuckDuckGo
      |
      +--> Tavily
      |
      v
 Final Answer

# Tech Stack

Python 3.10+

LangChain

LangGraph

ChromaDB

HuggingFace Embeddings

Groq LLM (qwen/qwen3-32b)

MCP (FastMCP, SSE client)

DuckDuckGo Search

Tavily API

# 📁 Project Structure
.
├── ingest_pdf.py    
# PDF loading, chunking, embedding, vector DB
├── mcp_server.py   
# MCP server exposing web tools
├── app.py  
# LangGraph RAG pipeline
├── chroma_db/    
# Persisted vector store
├── .env       
# API keys
└── README.md

# 🔐 Environment Variables

Create a .env file:

GROQ_API=your_groq_api_key

TAVILY_API=your_tavily_api_key

# PDF Ingestion

The PDF is:

GEN_AI Interview questions.pdf

What happens:

PDF is loaded using PyPDFLoader

Split into chunks (500 tokens, 50 overlap)

Embedded using all-MiniLM-L6-v2

Stored in ChromaDB (persistent)

db = function()  # returns Chroma vector store

# 🌐 MCP Web Tools
DuckDuckGo Tool

Used for:

Company info

General knowledge

Education & definitions

Tavily Tool

Used for:

Latest updates

News

Weather

Current events

Both are exposed via FastMCP server.

# 🧠 Intelligent Routing Logic
Routing Rules (Agent Node)
Question Type	Source
GenAI, LLMs, GANs, VAEs, Diffusion	PDF
Companies, definitions, history	DuckDuckGo
News, weather, latest updates	Tavily

The agent returns only one word:

pdf | duckduckgo | tavily

# 🔁 Confidence-Based PDF Check

Even if retrieved from PDF:

LLM judges relevance

If context is weak → fallback to web search

This avoids wrong PDF answers.

# 🧩 LangGraph Workflow

Entry point: pdf

Conditional routing based on:

PDF confidence

Agent decision

Final answer generated only from selected context

# ▶️ How to Run
# 1️⃣ Start MCP Server
python mcp_server.py

# 2️⃣ Run the App
python app.py

#3️⃣ Ask Questions
Ask a question (or exit): What is Generative AI?
Ask a question (or exit): What is Google?



