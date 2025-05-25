# 1) Python base
FROM python:3.11-slim

WORKDIR /app

# 2) Install Streamlit + requests + your requirements
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit requests \
    && pip install --no-cache-dir -r requirements.txt

# 3) Copy dashboard code
COPY dashboard/ ./dashboard/

# 4) Expose Streamlit’s default port
EXPOSE 8501

# 5) Run the Streamlit app
CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.address=0.0.0.0"]
