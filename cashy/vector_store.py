from __future__ import annotations
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

load_dotenv()

# initialize embedder
embedder = AzureOpenAIEmbeddings(
    deployment=os.environ["MODEL_EMBEDDINGS"],
    chunk_size=1,
)

# initialize vector store
vector_store = AzureSearch(
    azure_search_endpoint=os.environ["VECTOR_STORE_ADDRESS"],
    azure_search_key=os.environ["VECTOR_STORE_PASSWORD"],
    index_name=os.environ["VECTOR_STORE_ID"],
    embedding_function=embedder.embed_query,
)

# initialize vector store index client
azure_search_index_client = SearchIndexClient(
    os.environ["VECTOR_STORE_ADDRESS"],
    AzureKeyCredential(os.environ["VECTOR_STORE_PASSWORD"]),
)
