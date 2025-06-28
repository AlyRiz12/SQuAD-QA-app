FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 7860

# Run the Flask app via wsgi
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "wsgi:app"]
