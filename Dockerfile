FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl

RUN git config --global --add safe.directory /app

RUN python3 -m pip install --upgrade pip
RUN pip install poetry \
  && poetry config virtualenvs.create false
