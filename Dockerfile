FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face uses port 7860
EXPOSE 7860

# Start FastAPI server (IMPORTANT)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]