# Use an official TensorFlow runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install build dependencies for numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
# RUN pip install numpy==1.23.5 opencv-python tensorflow==2.12.0 streamlit==1.24.1
RUN pip3 install -r requirements.txt
RUN pip install matplotlib
# Make port 8501 available to the world outside this contain 
EXPOSE 8501

# Run the Streamlit app when the container launches
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
# CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
