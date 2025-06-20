# Use a base image with Python
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy your project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt wordnet omw-1.4

# Default command to run
CMD ["streamlit", "run", "search_app.py"]