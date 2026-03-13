# 🤖 Multi-Agent Document Q&A

A production-ready multi-agent system for intelligent document question answering, built with **LangChain**, **OpenAI**, and **ChromaDB**.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│         Orchestrator Agent              │  ← Routes, coordinates, combines
│      (GPT-4o + LangChain)               │
└──────┬──────────────┬───────────────────┘
       │              │
       ▼              ▼
┌─────────────┐  ┌───────────────┐
│  Retriever  │  │  Summarizer   │  ← Specialized sub-agents
│  Agent      │  │  Agent        │
│  (MMR)      │  │ (map-reduce)  │
└──────┬──────┘  └───────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
        ┌──────────────┐
        │ Critic Agent │  ← Validates accuracy & completeness
        └──────┬───────┘
               ▼
         Final Answer
```

### Agents

| Agent | Role | Key Feature |
|-------|------|-------------|
| **Orchestrator** | Routes queries, combines results | Multi-turn memory |
| **Retriever** | Semantic search over documents | MMR for diverse results |
| **Summarizer** | Condenses long text | Map-reduce for large docs |
| **Critic** | Validates & improves answers | Quality control layer |

## Features

- 📄 **Multi-format support** — PDF, TXT, DOCX, MD, CSV
- 🔍 **MMR retrieval** — Maximum Marginal Relevance for diverse, non-redundant results
- 🧠 **Conversation memory** — Multi-turn Q&A with full context
- 🎯 **Critic validation** — Every answer reviewed before delivery
- 💻 **Streamlit UI** — Clean web interface
- 🖥️ **CLI mode** — Terminal-based usage
- ✅ **Unit tests** — Pytest suite with mocks

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/multi-agent-doc-qa.git
cd multi-agent-doc-qa
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3a. Run Streamlit UI
streamlit run app.py

# 3b. Or use the CLI
export OPENAI_API_KEY=sk-...
python main.py --files docs/report.pdf docs/data.txt --model gpt-4o-mini
```

## Project Structure

```
multi-agent-doc-qa/
├── agents/
│   ├── orchestrator.py      # Main coordinator agent
│   ├── retriever_agent.py   # Semantic search agent
│   ├── summarizer_agent.py  # Summarization agent
│   └── critic_agent.py      # Answer validation agent
├── utils/
│   └── ingestion.py         # Document loading & embedding pipeline
├── tests/
│   └── test_agents.py       # Unit tests (pytest)
├── app.py                   # Streamlit web UI
├── main.py                  # CLI entrypoint
├── requirements.txt
└── .env.example
```

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `gpt-4o` | OpenAI model for agents |
| `chunk_size` | `1000` | Document chunk size (chars) |
| `chunk_overlap` | `200` | Overlap between chunks |
| `top_k` | `5` | Retrieved chunks per query |

## Tech Stack

- [LangChain](https://www.langchain.com/) — Agent framework
- [OpenAI GPT-4o](https://openai.com/) — LLM backbone
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [Streamlit](https://streamlit.io/) — Web UI

## License

MIT
