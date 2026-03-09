"""LLM client backed by the Anthropic API (Claude)."""
from __future__ import annotations
import logging
import os
import anthropic
from src.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Hellobooks AI, an expert AI assistant specialising in accounting, bookkeeping, and financial management for small and medium-sized businesses.

Guidelines:
- Answer ONLY based on the provided context. If the context does not contain enough information, say so honestly.
- Be concise, precise, and use plain English.
- When relevant, include specific numbers, formulas, or examples from the context.
- Structure longer answers with clear sections or bullet points.
- Indicate which topic area the answer relates to.
"""

RAG_PROMPT_TEMPLATE = """\
=== CONTEXT (retrieved from Hellobooks knowledge base) ===
{context}
==========================================================

USER QUESTION: {question}

Please answer the question based strictly on the context above.
"""

class LLMClient:
    def __init__(self, api_key=None, model=None):
        resolved_key = api_key or settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise EnvironmentError("Anthropic API key not found. Set ANTHROPIC_API_KEY in environment or .env file.")
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model or settings.llm_model

    def generate_answer(self, question, context_chunks):
        if not context_chunks:
            return ("I couldn't find relevant information in the Hellobooks knowledge base. "
                    "Please ask about bookkeeping, invoices, P&L, balance sheets, cash flow, or tax basics.")
        context_parts = []
        for rank, chunk in enumerate(context_chunks, start=1):
            doc = chunk["document"]
            score = chunk["score"]
            context_parts.append(f"[{rank}] Source: {doc.source.replace('_', ' ').title()} (relevance: {score:.2f})\n{doc.content}")
        context_text = "\n\n---\n\n".join(context_parts)
        user_message = RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)
        logger.debug("Calling %s with %d context chunks", self.model, len(context_chunks))
        message = self._client.messages.create(
            model=self.model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text
