# Use a stable Python image
FROM python:3.11-slim

# Make logs show up immediately
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Start your bot
CMD ["python", "main.py"]