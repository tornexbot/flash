# Use a stable Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of code
COPY . .

# Make sure environment variables are used
# (Render will provide TELEGRAM_TOKEN, WEBHOOK_URL, etc.)

# Start the bot
CMD ["python", "main.py"]
