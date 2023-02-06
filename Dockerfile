FROM python:3.9

# Copy files
COPY . /app

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install -r requirements/requirements.txt

# When training
# ENTRYPOINT ["python", "-u", "train.py"]

# When predicting
ENTRYPOINT ["python", "-u", "infer.py"]