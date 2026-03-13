"""
Retriever Agent - Handles semantic search over the document vectorstore
"""
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List


class RetrieverAgent:
    def __init__(self, vectorstore, llm: ChatOpenAI, top_k: int = 5):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.retriever: VectorStoreRetriever = vectorstore.as_retriever(
            search_type="mmr",  # Max Marginal Relevance for diversity
            search_kwargs={"k": top_k, "fetch_k": 20}
        )

    def retrieve(self, query: str) -> str:
        """Retrieve relevant document chunks for a query."""
        try:
            docs: List[Document] = self.retriever.invoke(query)

            if not docs:
                return "No relevant documents found for this query."

            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                results.append(
                    f"[Source {i}] File: {source} | Page: {page}\n"
                    f"Content: {doc.page_content[:800]}\n"
                )

            return "\n---\n".join(results)

        except Exception as e:
            return f"Retrieval error: {str(e)}"

    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """Retrieve docs with similarity scores (for debugging/evaluation)."""
        return self.vectorstore.similarity_search_with_score(query, k=self.top_k)
