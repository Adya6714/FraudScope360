# 1) Choose a lightweight Python base image
FROM python:3.11-slim

# 2) Set a working directory inside the container
WORKDIR /app

# 3) Copy only requirements first (for better caching)
COPY requirements.txt .

# 4) Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic \
    && pip install --no-cache-dir -r requirements.txt

# 5) Copy the API code into the image
COPY api/ ./api/
COPY features/ ./features/
COPY modules/ ./modules/
COPY configs/ ./configs/

# 6) Expose the port the API will listen on
EXPOSE 8000

# 7) Start the server with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]