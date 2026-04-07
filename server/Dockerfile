FROM python:3.11-slim

WORKDIR /DataDojo

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /DataDojo/server

EXPOSE 8080
CMD ["python","app.py","--port","8080"]