import os
from dotenv import load_dotenv #install library command (pip i dotenv)
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.tracing import set_tracing_disabled
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import logging
import frontmatter
from pathlib import Path
from typing import List

# Load environment variables from .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY") #get gemini api key from .env file

#Gemini API key not found condition
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file.")


# Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" #baseUrl from (Gemini Documention from Openai SDK docs)
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=provider,
)

set_tracing_disabled(disabled=True)