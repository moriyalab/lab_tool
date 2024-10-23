FROM python:3.10.15-slim-bullseye

RUN apt-get update && apt-get install -y git curl ffmpeg
RUN git config --global --add safe.directory /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install poetry \
  && poetry config virtualenvs.create false

WORKDIR /app
COPY pyproject.toml* poetry.lock* /app/
RUN poetry install 
RUN rm -rf /app/pyproject.toml* /app/poetry.lock*

# For Huggingface
# COPY . /app/

# CMD [ "python3", "lab_tool_webui.py" ]