FROM python:3.10.15-slim-bullseye AS builder_image

RUN apt-get update && apt-get install -y git curl ffmpeg
RUN git config --global --add safe.directory /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install poetry \
  && poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

FROM builder_image AS release_image

COPY --from=builder_image /app /app
COPY . /app/

CMD [ "python3", "app.py" ]