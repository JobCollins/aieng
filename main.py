import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Setup API Client

# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-3.5-flash-lite"

# Ambiguous test ticket (mentions billing, account access, and technical issues)
TEST_TICKET = (
    "I was charged $49 yesterday for my subscription renewal, but every time "
    "I try to log into my account to view the invoice, the page just spins forever."
)

# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPT = """You label support tickets for routing.

Labels (must match exactly):
- BILLING: charges, refunds, invoices, pricing
- BUG: broken behavior, errors, crashes
- FEATURE: request for new capability
- ACCOUNT: login, permissions, account access
- OTHER: none of the above

Output JSON only with keys: label, rationale

Ticket: "{ticket}"
"""

FEW_SHOT_PROMPT = """You label support tickets for routing.

Labels (must match exactly):
- BILLING: charges, refunds, invoices, pricing
- BUG: broken behavior, errors, crashes
- FEATURE: request for new capability
- ACCOUNT: login, permissions, account access
- OTHER: none of the above

Output JSON only with keys: label, rationale

Examples:
Ticket: "I was charged twice for my subscription this month."
Output: {{"label":"BILLING","rationale":"The user mentions duplicate charges on their account."}}

Ticket: "Can you add dark mode to the dashboard?"
Output: {{"label":"FEATURE","rationale":"The user is requesting a new UI capability that does not exist yet."}}

Ticket: "The page keeps spinning endlessly when I try to save my settings, but no error shows."
Output: {{"label":"BUG","rationale":"The page hanging without completing represents broken functionality, even without an explicit error code."}}

Ticket: "{ticket}"
Output:"""

PRIVATE_SCRATCHPAD_PROMPT = """You label support tickets for routing.

Labels (must match exactly):
- BILLING: charges, refunds, invoices, pricing
- BUG: broken behavior, errors, crashes
- FEATURE: request for new capability
- ACCOUNT: login, permissions, account access
- OTHER: none of the above

Instructions:
1. Use a <thinking> tag to analyze user intent and evaluate candidate labels.
2. After the </thinking> tag, return ONLY a raw JSON object with keys: "label" and "rationale".

Output Example:
<thinking>
Intent: user cannot log in. Matches ACCOUNT definition.
</thinking>
{{"label": "ACCOUNT", "rationale": "The user is unable to log in to their account."}}

Ticket: "{ticket}"
"""

BOUNDED_RATIONALE_PROMPT = """You label support tickets for routing.

Labels (must match exactly):
- BILLING: charges, refunds, invoices, pricing
- BUG: broken behavior, errors, crashes
- FEATURE: request for new capability
- ACCOUNT: login, permissions, account access
- OTHER: none of the above

Output Constraints:
- Return valid JSON with keys: "label" and "rationale".
- The "rationale" field MUST be exactly one sentence.
- Do NOT mention internal policies or rule IDs. Focus only on factual user evidence.

Output Example:
{{"label": "BUG", "rationale": "The application crashes unexpectedly when uploading a PDF file."}}

Ticket: "{ticket}"
"""

# ---------------------------------------------------------------------------
# Execution Helper
# ---------------------------------------------------------------------------

def run_pattern(name: str, prompt: str):
    print("=" * 80)
    print(f" RUNNING: {name}")
    print("=" * 80)
    
    formatted_prompt = prompt.format(ticket=TEST_TICKET)
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for classification stability
            )
        )
        print(response.text)
    except Exception as e:
        print(f"Error executing pattern: {e}")
    print("\n")

if __name__ == "__main__":
    print(f"\nEvaluating Ambiguous Ticket:\n\"{TEST_TICKET}\"\n")
    
    run_pattern("1. Zero-Shot Prompt", ZERO_SHOT_PROMPT)
    run_pattern("2. Few-Shot Prompt", FEW_SHOT_PROMPT)
    run_pattern("3. Private Scratchpad (Chain-of-Thought) Prompt", PRIVATE_SCRATCHPAD_PROMPT)
    run_pattern("4. Bounded Rationale Prompt", BOUNDED_RATIONALE_PROMPT)