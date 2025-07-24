# Use the specified base image and platform for AMD64 compatibility
# Using a "slim" image reduces the final image size.
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first. This leverages Docker's layer caching.
# If requirements.txt doesn't change, Docker won't re-install dependencies on subsequent builds.
COPY app/requirements.txt .

# Install the Python dependencies.
# --no-cache-dir ensures no cache is stored, keeping the image size down.
# --timeout increases the time pip will wait for a connection.
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

# Copy the rest of your application code, including the models directory.
# This will copy the entire 'app/' folder contents into the '/app' WORKDIR.
COPY app/ .

# This is the command that will be executed when the container starts.
# It runs your Python script to process the PDFs.
CMD ["python", "process_pdfs.py"]
