# Use an official TensorFlow runtime as a parent image

# FROM docker.io/tensorflow/tensorflow:nightly
FROM tensorflow/tensorflow:latest
# FROM http://docker.io/tensorflow/tensorflow:nightly
# FROM tensorflow/tensorflow:nightly
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]



# FROM python:3.9-slim
# # FROM tensorflow/tensorflow:nightly


# WORKDIR /app

# # RUN apt-get update && apt-get install -y ca-certificates\

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# COPY . .

# # RUN pip3 install -r requirements.txt
# RUN python -m pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org tensorflow

# RUN python -m pip install -r requirements.txt

# # RUN python -m pip install streamlit
# # RUN python -m pip install opencv-python
# # RUN python -m pip install tensorflow 
 


# # RUN pip3 install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host=files.pythonhosted.org -r requirements.txt


# EXPOSE 8080

# HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

# # ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]