# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# The app uses standard library only, so no requirements to install
# However, we'll keep the requirements.txt structure for future use
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]
