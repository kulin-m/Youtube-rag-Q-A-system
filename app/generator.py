"""
app/generator.py
LLM Generation via Groq API.
"""
import logging
import time
from typing import Optional

from groq import Groq

from config.config import cfg
from app.utils import (
    build_fid_style_contextual_prompt,
    build_fid_style_generalized_prompt,
)

logger = logging.getLogger(__name__)

def _discover_best_model(client: Groq) -> str:
    try:
        preferred_models = [
            "llama-3.3-70b-versatile", 
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
        return preferred_models[0]
    except Exception as e:
        logger.warning(f"[Generator] Model selection failed: {e}")
    return "llama-3.3-70b-versatile"

class GroqGenerator:
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = api_key or cfg.GROQ_API_KEY 
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = _discover_best_model(self.client)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"[Generator] Using Groq model: {self.model_name}")

    def _call_groq(self, prompt: str) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=0.3,
                    max_tokens=2048,
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                if "rate_limit_exceeded" in str(e).lower():
                    wait_time = self.retry_delay * (attempt ** 2)
                    logger.warning(f"[Generator] Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"[Generator] Attempt {attempt} failed: {e}")
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)
                    else:
                        raise RuntimeError(f"Groq API failed after {self.max_retries} attempts: {e}")

    def generate_contextual_answer(self, question: str, context_chunks: list, source_titles: list = None, source_timestamps: list = None) -> str:
        if not context_chunks:
            return "I could not find relevant information in the video transcript(s)."
        
        prompt = build_fid_style_contextual_prompt(question, context_chunks, source_titles, source_timestamps)
        logger.debug("[Generator] Generating contextual answer via Groq...")
        return self._call_groq(prompt)

    def generate_generalized_answer(self, question: str, context_chunks: list, source_titles: list = None) -> str:
        prompt = build_fid_style_generalized_prompt(question, context_chunks, source_titles)
        logger.debug("[Generator] Generating generalized answer via Groq...")
        return self._call_groq(prompt)