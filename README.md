# hia-chatbot

Chat with [121 user manual](https://manual.121.global).
 
## Description

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API to serve a chatbot for [121](https://github.com/global-121). Based on [langchain](https://github.com/langchain-ai/langchain) and OpenAI models. Uses [Poetry](https://python-poetry.org/) for dependency management.

## API Usage

See [the docs](https://hia-chatbot.azurewebsites.net/docs).

### Configuration

```sh
cp example.env .env
```

Edit the provided [ENV-variables](./example.env) accordingly.

### Run locally

```sh
pip install poetry
poetry install
uvicorn main:app --reload
```

### Run with Docker

```sh
docker compose up --detach
```

