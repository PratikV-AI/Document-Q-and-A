"""
Streamlit UI for Multi-Agent Document Q&A
"""
import os
import tempfile
import streamlit as st
from pathlib import Path

from utils.ingestion import ingest_documents, load_vectorstore
from agents.orchestrator import OrchestratorAgent

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Doc Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model = st.selectbox("LLM Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    verbose = st.checkbox("Verbose agent logs", value=False)

    st.divider()
    st.markdown("### 📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, or MD files",
        type=["pdf", "txt", "docx", "md", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            with st.spinner("Ingesting documents..."):
                try:
                    # Save uploaded files to temp directory
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []
                    for f in uploaded_files:
                        path = os.path.join(temp_dir, f.name)
                        with open(path, "wb") as out:
                            out.write(f.read())
                        file_paths.append(path)

                    # Run ingestion pipeline
                    vectorstore = ingest_documents(file_paths)

                    # Initialize orchestrator
                    st.session_state.orchestrator = OrchestratorAgent(
                        vectorstore=vectorstore,
                        llm_model=model,
                        verbose=verbose
                    )
                    st.session_state.docs_loaded = True
                    st.success(f"✅ Processed {len(uploaded_files)} document(s)!")

                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.docs_loaded:
        st.success("📚 Documents ready!")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            if st.session_state.orchestrator:
                st.session_state.orchestrator.reset_memory()
            st.rerun()

    st.divider()
    st.markdown("""
    ### 🤖 Agent Pipeline
    1. **Orchestrator** — routes query
    2. **Retriever** — semantic search (MMR)
    3. **Summarizer** — condenses content
    4. **Critic** — validates final answer
    """)

# ─── Main UI ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🤖 Multi-Agent Document Q&A</div>', unsafe_allow_html=True)
st.caption("Powered by LangChain · OpenAI · ChromaDB · 4-agent pipeline")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.docs_loaded:
        st.warning("⬅️ Please upload and process documents first.")
    elif not api_key:
        st.warning("⬅️ Please enter your OpenAI API key.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Agents are thinking..."):
                try:
                    result = st.session_state.orchestrator.run(prompt)
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = f"❌ Error: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

if not st.session_state.docs_loaded:
    st.info("👈 Upload documents in the sidebar to get started.")
