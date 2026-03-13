"""
Summarizer Agent - Condenses long document sections into clear answers
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document


SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["text", "instruction"],
    template="""You are a precise document summarizer.

Instruction: {instruction}

Text to summarize:
{text}

Provide a clear, structured summary that directly addresses the instruction.
Use bullet points where appropriate. Be concise but complete."""
)


class SummarizerAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def summarize(self, input_text: str) -> str:
        """
        Summarize text with optional instructions.
        Input format: 'INSTRUCTION: <instruction>\nTEXT: <text>'
        Or just plain text for a general summary.
        """
        try:
            instruction = "Provide a concise and accurate summary."
            text = input_text

            # Parse structured input if provided
            if "INSTRUCTION:" in input_text and "TEXT:" in input_text:
                parts = input_text.split("TEXT:", 1)
                instruction = parts[0].replace("INSTRUCTION:", "").strip()
                text = parts[1].strip()

            # Truncate if too long (avoid token limits)
            max_chars = 12000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]"

            prompt = SUMMARIZE_PROMPT.format(text=text, instruction=instruction)
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            return f"Summarization error: {str(e)}"

    def map_reduce_summarize(self, docs: list[Document]) -> str:
        """Use map-reduce for very long documents."""
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        result = chain.invoke({"input_documents": docs})
        return result["output_text"]
