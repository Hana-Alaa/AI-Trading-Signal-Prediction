FROM python:3.12.4-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "api.phase8_paper_trading_api:app", "--host", "0.0.0.0", "--port", "8000"]