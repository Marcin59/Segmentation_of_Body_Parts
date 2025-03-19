FROM python:3.11-slim

WORKDIR /app

COPY app.py /app/app.py
COPY data/models/final_model.keras /app/data/models/final_model.keras
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port", "8080"]
