"""
LLM Client

This module is the client boundary configuration for the LLM client. 

"""

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv()

@dataclass
class LLMResponse:
    """
    Response from an LLM provider.
    """
    provider: str
    text: str
    usage: dict
    raw: Any

def require_env(name: str) -> str:
    """
    Require an environment variable to be set.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


def call_gemini(prompt:str, model="gemini-3.6-flash") -> LLMResponse:
    """
    Provide call function for Gemini.
    """
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    text = response.text or ""
    usage = {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count,
    } if response.usage_metadata else {}
    return LLMResponse("gemini", text, usage, response)