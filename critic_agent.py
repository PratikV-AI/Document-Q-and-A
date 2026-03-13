"""
Critic Agent - Validates and improves answers for accuracy and completeness
"""
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


CRITIC_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are a critical evaluator for AI-generated answers about documents.

Original Question: {question}

Proposed Answer: {answer}

Evaluate this answer on:
1. **Accuracy** - Is it factually correct based on what was stated?
2. **Completeness** - Does it fully address the question?
3. **Clarity** - Is it well-structured and easy to understand?
4. **Citations** - Are sources mentioned where appropriate?

If the answer is good, return it as-is with a brief validation note.
If it needs improvement, provide an improved version.

Respond in this format:
VERDICT: [APPROVED / IMPROVED]
REASONING: <one sentence>
FINAL_ANSWER: <the final answer to use>"""
)


class CriticAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def validate(self, input_text: str) -> str:
        """
        Validate and potentially improve an answer.
        Input: JSON string with 'question' and 'answer' keys,
               or plain 'question|||answer' format.
        """
        try:
            # Try JSON parsing first
            try:
                data = json.loads(input_text)
                question = data.get("question", "")
                answer = data.get("answer", "")
            except json.JSONDecodeError:
                # Fallback: split by delimiter
                if "|||" in input_text:
                    parts = input_text.split("|||", 1)
                    question = parts[0].strip()
                    answer = parts[1].strip()
                else:
                    # Can't parse — just return the input
                    return input_text

            if not question or not answer:
                return answer or input_text

            prompt = CRITIC_PROMPT.format(question=question, answer=answer)
            response = self.llm.invoke(prompt)
            content = response.content

            # Extract final answer from structured response
            if "FINAL_ANSWER:" in content:
                final = content.split("FINAL_ANSWER:", 1)[1].strip()
                return final

            return content

        except Exception as e:
            return f"Critic error: {str(e)}\nOriginal answer: {input_text}"
