import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

#Gemini API key not found condition
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file.")

# Provider - Using Google's Gemini via OpenAI-compatible endpoint
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Google's OpenAI-compatible endpoint
)

# Model name to be used by the agent
model = "gemini-2.5-flash"