# Base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirement file first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]