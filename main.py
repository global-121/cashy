from __future__ import annotations
import uvicorn
from langchain_core.prompts import PromptTemplate
from fastapi import (
    Security,
    Depends,
    FastAPI,
    APIRouter,
    Request,
    HTTPException,
    Header,
)
from langchain_community.document_loaders.sitemap import SitemapLoader
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.RAG.vector_store import VectorStore
from src.constants import DocumentMetadata

dm = DocumentMetadata()
from time import perf_counter
import math
import statistics
import os
import re
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
from dotenv import load_dotenv

load_dotenv()

# load environment variables
if "PORT" not in os.environ.keys():
    port = 8000
else:
    port = os.environ["PORT"]

description = """
Chat with [121 user manual](https://manual.121.global/) 💬🤖

Built with love by [NLRC 510](https://www.510.global/). See
[the project on GitHub](https://github.com/global-121) or [contact us](mailto:support@510.global).
"""

# initialize FastAPI
app = FastAPI(
    title="121-chatbot",
    description=description,
    version="0.0.1",
    license_info={
        "name": "AGPL-3.0 license",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
key_query_scheme = APIKeyHeader(name="Authorization")

azure_search_index_client = SearchIndexClient(
    os.environ["VECTOR_STORE_ADDRESS"],
    AzureKeyCredential(os.environ["VECTOR_STORE_PASSWORD"]),
)

vector_store_id = "121chatbot"


@app.get("/", include_in_schema=False)
async def docs_redirect():
    """Redirect base URL to docs."""
    return RedirectResponse(url="/docs")


class QuestionPayload(BaseModel):
    question: str = Field(
        ...,
        description="""
        Text of the question""",
    )


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


@app.post("/ask")
async def ask_question(
    payload: QuestionPayload, api_key: str = Depends(key_query_scheme)
):
    """Ask something to the chatbot and get an answer."""

    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # initialize vector store
    t2_start = perf_counter()
    vector_db = VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_service="azuresearch",
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        embedding_source="OpenAI",
        embedding_model=os.environ["MODEL_EMBEDDINGS"],
        store_id=vector_store_id,
    )
    t2_stop = perf_counter()
    logger.info(
        f"Elapsed time getting vector store: {float(t2_stop - t2_start)} seconds"
    )

    # create the retriever and the QA pipeline
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["MODEL_CHAT"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    # prompt = hub.pull("rlm/rag-prompt")

    template = """You are a chatbot that gives an answer to a question. Answer the question truthfully based solely on
     the given documents. If you don't know the answer, just say that you don't know, don't try to make up an answer.
     If no documents are provided to answer the question, answer exactly this: 'I don't have the right information
     to answer your question'.
     Documents: {context}
     Question: {question}
     Answer:"""
    prompt = PromptTemplate.from_template(template)

    retrieved_docs = vector_db.langchain_client.similarity_search(payload.question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt_invocation = prompt.invoke(
        {"question": payload.question, "context": docs_content}
    )
    answer = llm.invoke(prompt_invocation)
    t2_stop = perf_counter()
    logger.info(f"Elapsed time getting answer: {float(t2_stop - t2_start)} seconds")

    return answer


# class VectorStorePayload(BaseModel):
#     url: str = Field(
#         ...,
#         description="""
#     URL to user manual""",
#     )


@app.post("/create-vector-store")
async def create_vector_store(
    api_key: str = Depends(key_query_scheme),  # , payload: VectorStorePayload
):
    """Create a vector store from a HIA instance. Replace all entries if it already exists."""

    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # load documents from Google Sheet
    doc_loader = SitemapLoader(
        web_path="https://manual.121.global/en/sitemap.xml",
        filter_urls=[
            "https://manual.121.global/faq/",
            "https://manual.121.global/general/",
            "https://manual.121.global/monitoring/",
            "https://manual.121.global/payment/",
            "https://manual.121.global/registration/",
            "https://manual.121.global/team/",
            "https://manual.121.global/users/",
            "https://manual.121.global/verification/",
        ],
    )
    docs = doc_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    docs = text_splitter.split_documents(docs)

    # add documents to vector store
    vector_store = VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_service="azuresearch",
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        embedding_source="OpenAI",
        embedding_model=os.environ["MODEL_EMBEDDINGS"],
        store_id=vector_store_id,
    )
    n_docs = vector_store.add_documents(docs)

    return JSONResponse(
        status_code=200,
        content=f"Created index {vector_store_id} with {n_docs} documents.",
    )


@app.get("/get-models")
async def get_models_used_in_chatbot():
    """Get models used in the chatbot."""
    return JSONResponse(
        status_code=200,
        content={
            "endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "chatbot": os.environ["MODEL_QA"],
            "embeddings": os.environ["MODEL_EMBEDDINGS"],
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=True)
