"""
Unit tests for Multi-Agent Document Q&A system
Run: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from agents.retriever_agent import RetrieverAgent
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mocked LLM response")
    return llm


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    vs.as_retriever.return_value.invoke.return_value = [
        Document(
            page_content="The capital of France is Paris.",
            metadata={"source": "geography.pdf", "page": 1}
        ),
        Document(
            page_content="Paris is known as the City of Light.",
            metadata={"source": "geography.pdf", "page": 2}
        )
    ]
    return vs


# ─── RetrieverAgent Tests ────────────────────────────────────────────────────

class TestRetrieverAgent:
    def test_retrieve_returns_formatted_results(self, mock_vectorstore, mock_llm):
        agent = RetrieverAgent(mock_vectorstore, mock_llm)
        result = agent.retrieve("What is the capital of France?")
        assert "Paris" in result
        assert "geography.pdf" in result
        assert "Source 1" in result

    def test_retrieve_handles_empty_results(self, mock_vectorstore, mock_llm):
        mock_vectorstore.as_retriever.return_value.invoke.return_value = []
        agent = RetrieverAgent(mock_vectorstore, mock_llm)
        result = agent.retrieve("query with no results")
        assert "No relevant documents found" in result

    def test_retrieve_handles_exception(self, mock_vectorstore, mock_llm):
        mock_vectorstore.as_retriever.return_value.invoke.side_effect = Exception("DB error")
        agent = RetrieverAgent(mock_vectorstore, mock_llm)
        result = agent.retrieve("test query")
        assert "Retrieval error" in result


# ─── SummarizerAgent Tests ────────────────────────────────────────────────────

class TestSummarizerAgent:
    def test_summarize_plain_text(self, mock_llm):
        agent = SummarizerAgent(mock_llm)
        result = agent.summarize("This is some text to summarize.")
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()

    def test_summarize_with_instruction(self, mock_llm):
        agent = SummarizerAgent(mock_llm)
        result = agent.summarize(
            "INSTRUCTION: Focus on key dates\nTEXT: The company was founded in 1990."
        )
        assert result == "Mocked LLM response"

    def test_summarize_truncates_long_text(self, mock_llm):
        agent = SummarizerAgent(mock_llm)
        long_text = "word " * 5000  # Very long text
        result = agent.summarize(long_text)
        # Verify it called llm (didn't crash)
        mock_llm.invoke.assert_called_once()

    def test_summarize_handles_exception(self, mock_llm):
        mock_llm.invoke.side_effect = Exception("API error")
        agent = SummarizerAgent(mock_llm)
        result = agent.summarize("Test text")
        assert "Summarization error" in result


# ─── CriticAgent Tests ────────────────────────────────────────────────────────

class TestCriticAgent:
    def test_validate_json_input(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="VERDICT: APPROVED\nREASONING: Good\nFINAL_ANSWER: Paris is the capital of France."
        )
        agent = CriticAgent(mock_llm)
        result = agent.validate('{"question": "Capital of France?", "answer": "Paris"}')
        assert "Paris" in result

    def test_validate_pipe_delimited_input(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="FINAL_ANSWER: Validated answer here."
        )
        agent = CriticAgent(mock_llm)
        result = agent.validate("What is 2+2?|||The answer is 4.")
        assert "Validated answer here." in result

    def test_validate_returns_original_on_bad_input(self, mock_llm):
        agent = CriticAgent(mock_llm)
        result = agent.validate("no special format here at all")
        # Should return original without crashing
        assert isinstance(result, str)

    def test_validate_handles_exception(self, mock_llm):
        mock_llm.invoke.side_effect = Exception("API error")
        agent = CriticAgent(mock_llm)
        result = agent.validate('{"question": "Q", "answer": "A"}')
        assert "Critic error" in result


# ─── Integration-style test ───────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_retrieve_then_summarize(self, mock_vectorstore, mock_llm):
        retriever = RetrieverAgent(mock_vectorstore, mock_llm)
        summarizer = SummarizerAgent(mock_llm)

        retrieved = retriever.retrieve("Tell me about Paris")
        assert "Paris" in retrieved

        summary = summarizer.summarize(f"INSTRUCTION: Summarize briefly\nTEXT: {retrieved}")
        assert summary == "Mocked LLM response"
