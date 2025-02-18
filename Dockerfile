# first stage
#FROM python:3.12 AS builder
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./detection /detection

#COPY ./detection/requirements.txt ./detection/requirements.txt

RUN pip install --no-cache-dir -r /detection/requirements.txt

#RUN pip install -q git+https://github.com/huggingface/transformers.git

#RUN pip install "fastapi[standart]"

WORKDIR /detection

#CMD ["fastapi", "run", "server.py", "--port", "80"]

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]