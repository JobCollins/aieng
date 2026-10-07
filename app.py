from llm_client import call_gemini

if __name__ == "__main__":
    prompt = "Say hi in five words."
    result = call_gemini(prompt)
    print(f"[{result.provider}] text: {result.text[:80]!r}") #!r is for raw string representation
    print(f"[{result.provider}] usage: {result.usage}")