"""
CLI interface for Multi-Agent Document Q&A
Usage: python main.py --files doc1.pdf doc2.txt
"""
import os
import argparse
from utils.ingestion import ingest_documents, load_vectorstore
from agents.orchestrator import OrchestratorAgent


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Document Q&A")
    parser.add_argument("--files", nargs="+", help="Paths to documents to ingest")
    parser.add_argument("--load-db", default="./chroma_db", help="Load existing vectorstore")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent output")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    # Load or create vectorstore
    if args.files:
        vectorstore = ingest_documents(args.files)
    else:
        print(f"Loading existing vectorstore from {args.load_db}...")
        vectorstore = load_vectorstore(args.load_db)

    # Initialize orchestrator
    orchestrator = OrchestratorAgent(
        vectorstore=vectorstore,
        llm_model=args.model,
        verbose=args.verbose
    )

    print("\n🤖 Multi-Agent Document Q&A ready!")
    print("Type 'quit' or 'exit' to stop. Type 'reset' to clear conversation history.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if query.lower() == "reset":
            orchestrator.reset_memory()
            print("✅ Conversation history cleared.\n")
            continue

        result = orchestrator.run(query)
        print(f"\nAssistant: {result['answer']}\n")


if __name__ == "__main__":
    main()
