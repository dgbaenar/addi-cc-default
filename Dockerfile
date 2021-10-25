FROM python:3.8

RUN apt update && apt install python-dev -y

WORKDIR /app
COPY requirements.txt /app
COPY env.yml /app
COPY src/. /app
COPY data /app/data

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000
