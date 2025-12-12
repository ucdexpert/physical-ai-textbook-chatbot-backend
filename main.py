"""
Vercel entry point for the Physical AI RAG API
This file serves as the entry point for Vercel deployment
"""
from api.index import app

# The Vercel Python runtime expects the FastAPI app to be available at the module level
# This allows Vercel to properly handle the application
app_instance = app

# For local development, you can run this file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)