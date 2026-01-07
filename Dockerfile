# -------- Base image --------
FROM python:3.12.4-slim

# -------- Environment --------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------- Working directory --------
WORKDIR /app

# -------- Python deps --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- Copy project --------
COPY . .

# -------- Expose API port --------
EXPOSE 8000

# -------- Run API --------
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
