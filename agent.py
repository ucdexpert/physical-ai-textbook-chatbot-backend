import os
from agents.agent import Agent
from tools import search_book_content, format_context_for_answer
from gemini_config import model

# Define the instructions for the agent behavior
SYSTEM_INSTRUCTIONS = """
You are an expert tutor for a Physical AI & Humanoid Robotics textbook.
Your goal is to answer student questions accurately using ONLY the textbook content.

PROTOCOL:
1. ALWAYS uses the 'search_book_content' tool to find information about the user's query.
2. ALWAYS use the 'format_context_for_answer' tool to structure the search results.
3. Answer the user's question based on the formatted context.
4. Cite the Chapter and Section for every fact you mention.
5. If the search returns no results, state clearly that the information is not in the book.
"""

# Instantiate the Agent
textbook_agent = Agent(
    name="Physical AI Tutor",
    instructions=SYSTEM_INSTRUCTIONS,
    tools=[search_book_content, format_context_for_answer],
    model=model 
)