FROM python:3.9

# Copy files
COPY . /app

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install -r requirements/requirements.txt


# Run the application
EXPOSE 8000
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000"]
CMD ["app:app"]