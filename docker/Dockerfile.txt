FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY transformadores/ ./transformadores/
COPY inference.py pipeline.pkl ./

CMD [ "python", "./inference.py" ]
