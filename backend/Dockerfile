FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the container's working directory
COPY . .

# Install dependencies from the requirements file without using cache to reduce image size
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Updated command using `--preload` to load joblib files ONCE before workers spawn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "--timeout", "120", "--preload", "app:product_selling_predictor_api"]
