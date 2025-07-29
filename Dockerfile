# Use a lightweight, amd64-compatible Python image 
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements first to leverage caching
COPY requirements.txt .

# Install dependencies without using cache or accessing internet (offline requirement)
RUN pip install --no-cache-dir --no-index --find-links=./wheels -r requirements.txt || true

# Copy the rest of the application code
COPY . .

# Download and save the sentence transformer model locally
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('all-MiniLM-L6-v2-local')"

# Ensure output directory exists
RUN mkdir -p /app/Output

# Entrypoint: process all PDFs in /app/input and save results to /app/output
CMD ["python", "1B.py", "Input"]
