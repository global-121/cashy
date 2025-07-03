import json
import os
import logging
from src.constants import LOGGING_LEVEL
import uuid
import tiktoken
from langchain_openai import AzureChatOpenAI
from src.constants import LOGGING_LEVEL, AzureChatMetadata

llmm = AzureChatMetadata()


def uuid_hash(content: str) -> str:
    """Create a unique hash from the page content of a document"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))


def initialize_logger(name):
    logging.basicConfig(
        level=LOGGING_LEVEL,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(name)

    return logger


def set_openai_env_vars(path_to_openai_config: str):
    """
    Based on an config JSON file, set the environment variables for the OpenAI API
    """
    with open(path_to_openai_config, "rt", encoding="utf-8") as f:
        openai_config = json.load(f)

    for k, v in openai_config.items():
        os.environ[k] = v


def set_llm(temperature: float) -> AzureChatOpenAI:
    """
    setting the LLM to generate the synthetic dataset with
    """

    llm = AzureChatOpenAI(
        deployment_name=llmm.DEPLOYMENT_NAME,
        model_name=llmm.MODEL_NAME,
        openai_api_version=llmm.OPENAI_API_VERSION,
        temperature=temperature,
    )

    return llm


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
