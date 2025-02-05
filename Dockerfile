# first stage
FROM python:3.12 AS builder

COPY ./detection_test/requirements.txt .

# install dependencies to the local user directory (eg. /root/.local)
RUN pip install --user --no-cache-dir -r requirements.txt

# second stage
FROM python:3.12-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# copy only the dependencies that are needed for our application and the source files
COPY --from=builder /root/.local /root/.local

COPY ./detection_test /detection_test

WORKDIR /detection_test


# update PATH
ENV PATH=/root/.local:$PATH

# make sure you include the -u flag to have our stdout logged
CMD [ "python3", "-u", "./main.py" ]