import json
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from google import genai
from google.genai import types

import os
from dotenv import load_dotenv

load_dotenv()

# Setup API Client 
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-3.5-flash-lite"

# Sample support ticket input
TEST_TICKET = (
    "I'm unable to log into my account since this morning. "
    "The submit button is grayed out and throwing a 500 server error."
)

# ---------------------------------------------------------------------------
# 1. Define the pydantic schema (validation & Enums)
# ---------------------------------------------------------------------------

class UrgencyEnum(str, Enum):
    """
    The urgency of the ticket
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CategoryEnum(str, Enum):
    """
    The category of the ticket
    """
    ACCOUNT = "account"
    BILLING = "billing"
    FEATURE = "feature"
    BUG = "bug"
    OTHER = "other"

class CustomerInfo(BaseModel):
    """
    Optional customer information
    """
    email: Optional[str] = Field(default=None, description="The email address of the customer")
    account_id: Optional[str] = Field(default=None, description="The account ID of the customer")

class SupportTicketOutput(BaseModel):
    """
    The output of the support ticket
    """
    category: CategoryEnum = Field(description="The category of the ticket")
    urgency: UrgencyEnum = Field(description="The urgency of the ticket")
    summary: str = Field(description="A short summary of the issue")
    customer: Optional[CustomerInfo] = Field(default=None, description="Optional customer information")

# ---------------------------------------------------------------------------
# 2. Pipeline execution function with pydantic validation and repair loop
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a support ticket classification assistant.
Return ONLY valid JSON matching the specified structure.
Do not add commentary, explanations, or markdown code fences."""

def process_ticket(ticket_text: str, max_retries: int = 2) -> Optional[SupportTicketOutput]:
    print(f"Processing Ticket:\n\"{ticket_text}\"\n")

    current_prompt = f"Task:\n{ticket_text}\n\nSchema:\n{json.dumps(SupportTicketOutput.model_json_schema(), indent=2)}"

    for attempt in range(1, max_retries +2): # +2 because we want to include the initial attempt and the repair attempt
        print(f"---------Attempt {attempt} --------------------------------")
        try:
            # Call Gemini API enforcing JSON response structure via response_schema
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=SupportTicketOutput
                )
            )

            raw_output = response.text
            print(f"Raw output:\n{raw_output}\n")

            # Parse the JSON response
            parsed_data = json.loads(raw_output)

            # Validate the parsed data using pydantic into typed object
            validated_object = SupportTicketOutput.model_validate(parsed_data)
            print("Successfully validated into Pydantic model!")
            print(f"Parsed Category: {validated_object.category.value}")
            print(f"Parsed Urgency:  {validated_object.urgency.value}")
            print(f"Parsed Summary:  {validated_object.summary}\n")
            return validated_object
        except (json.JSONDecodeError, ValidationError) as e:
            error_message = str(e)
            print(f"Error Validation/Parsing failed {error_message}\n")

            if attempt > max_retries:
                print(f"Max retries reached. Final output:\n{raw_output}\n")
                return None
            
            # construct a repair prompt
            print("Constructing repair prompt...")
            current_prompt = (
                f"Your previous output failed validation.\n"
                f"Error details: {error_message}\n"
                f"Previous output: {raw_output}\n"
                f"Please fix the errors and return valid JSON strictly matching the specified schema.\n"
            )

if __name__ == "__main__":
    result = process_ticket(TEST_TICKET)