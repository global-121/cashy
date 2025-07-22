from __future__ import annotations
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    Request,
    HTTPException,
)
from langchain_community.document_transformers import MarkdownifyTransformer
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from cashy.vector_store import vector_store, azure_search_index_client
from cashy.agent import agent_graph
import os
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
    title="💸 Cashy",
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


@app.post("/ask")
async def ask_question(
    payload: QuestionPayload, request: Request, api_key: str = Depends(key_query_scheme)
):
    """Ask something to the chatbot and get an answer."""

    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # use client host as memory thread ID
    client_host = request.client.host
    config = {"configurable": {"thread_id": client_host}}

    # invoke the agent graph with the question
    response = agent_graph.invoke(
        {"messages": [{"role": "user", "content": payload.question}]}, config=config
    )
    answer = response["messages"][-1].content

    return answer


@app.post("/update-vector-store")
async def update_vector_store(
    api_key: str = Depends(key_query_scheme),
):
    """Create a vector store from the 121 manual. Replaces the existing vector store with a new one."""

    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # load documents from 121 user manual
    doc_loader = RecursiveUrlLoader(
        "https://manual.121.global/en/",
        prevent_outside=True,
        base_url="https://manual.121.global/en/",
        exclude_dirs=["https://manual.121.global/en/nlrc"],
    )
    docs = doc_loader.load()

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    docs = text_splitter.split_documents(docs)

    # convert HTML to markdown
    md = MarkdownifyTransformer(strip="a")
    docs = md.transform_documents(docs)

    # delete existing data
    azure_search_index_client.delete_index(os.environ["VECTOR_STORE_ID"])

    # embed docs and store to vector db
    vector_store.add_documents(docs)

    return JSONResponse(
        status_code=200,
        content=f"Updated {os.environ['VECTOR_STORE_ID']} with {len(docs)} documents.",
    )


@app.get("/get-models")
async def get_models():
    """Get the models used."""
    return JSONResponse(
        status_code=200,
        content={
            "chatbot": os.environ["MODEL_CHAT"],
            "embeddings": os.environ["MODEL_EMBEDDINGS"],
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=True)
