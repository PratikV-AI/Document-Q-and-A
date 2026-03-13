"""
Orchestrator Agent - Routes queries to specialized sub-agents
"""
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from agents.retriever_agent import RetrieverAgent
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent


ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent orchestrator that manages a team of specialized AI agents to answer questions about documents.

You have access to the following agents:
1. **Retriever Agent** - Finds relevant chunks from uploaded documents using semantic search
2. **Summarizer Agent** - Summarizes long documents or sections into concise answers
3. **Critic Agent** - Reviews and validates answers for accuracy and completeness

Your job:
- Understand the user's query
- Route to the appropriate agent(s)
- Combine results into a coherent, well-structured final answer
- Always cite the source document and page number when possible

Be concise, factual, and helpful. If you're unsure, say so clearly."""


class OrchestratorAgent:
    def __init__(self, vectorstore, llm_model: str = "gpt-4o", verbose: bool = True):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.vectorstore = vectorstore
        self.verbose = verbose

        # Initialize sub-agents
        self.retriever = RetrieverAgent(vectorstore, self.llm)
        self.summarizer = SummarizerAgent(self.llm)
        self.critic = CriticAgent(self.llm)

        # Build tools from sub-agents
        self.tools = self._build_tools()

        # Memory for multi-turn conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Build the orchestrator agent
        self.agent_executor = self._build_agent()

    def _build_tools(self):
        return [
            Tool(
                name="retrieve_documents",
                func=self.retriever.retrieve,
                description=(
                    "Use this to search and retrieve relevant text chunks from the uploaded documents. "
                    "Input should be a specific search query. Returns relevant document passages with sources."
                )
            ),
            Tool(
                name="summarize_content",
                func=self.summarizer.summarize,
                description=(
                    "Use this to summarize long pieces of text into a concise response. "
                    "Input should be the text you want summarized plus any specific instructions."
                )
            ),
            Tool(
                name="validate_answer",
                func=self.critic.validate,
                description=(
                    "Use this to validate and improve an answer before returning it. "
                    "Input should be a JSON string with 'question' and 'answer' keys."
                )
            ),
        ]

    def _build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", ORCHESTRATOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def run(self, query: str) -> dict:
        """Run a query through the multi-agent pipeline."""
        result = self.agent_executor.invoke({"input": query})
        return {
            "query": query,
            "answer": result["output"],
            "chat_history": self.memory.chat_memory.messages
        }

    def reset_memory(self):
        """Clear conversation history."""
        self.memory.clear()
