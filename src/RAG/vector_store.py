import copy
from pathlib import Path
from typing import List

from chromadb import PersistentClient
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.azuresearch import AzureSearch
from src.constants import DocumentMetadata
from src.RAG.utils import initialize_logger, uuid_hash
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswParameters,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

DEFAULT_HUGGING_FACE_MODEL = "sentence-transformers/all-mpnet-base-v2"


dm = DocumentMetadata()


class VectorStore:
    """
    Vector storage for chunked documents and embeddings
        1. Embeds chunked documents
        2. Save to or loads from path
        3. Optionally embeds and adds new chunked documents to existing DB

    Input
    --------
    store_path:
        str
        path to store the vectors
    store_service:
        str
        name of the store service, currently supports 'chroma' and 'azuresearch'
    store_password:
        str
        password for the store service (e.g. for Azure Search)
    embedding_source:
        str
        name of the source of the embedding model
    embedding_model:
        str
        name of the embedding model
    store_id:
        str
        id of the collection (Chroma) or index (Azure Search)
    chunked_documents:
        List[Document]
        list of chunked Langchain Documents with metadata
    """

    output: List[Document]

    def __init__(
        self,
        store_path: str,
        store_service: str,
        store_password: str = None,
        embedding_source: str = None,
        embedding_model: str = None,
        store_id: str = "chunked_document_embeddings",
    ):
        self.logger = initialize_logger(self.__class__.__name__)
        self.store_id = store_id
        self.embedding_source = embedding_source
        self.embedding_model = embedding_model
        self.store_service = store_service
        self.store_password = store_password
        self.store_path = store_path
        self.embedder = self._set_embedder()
        self.client = self._set_client()
        self.langchain_client = self._set_langchain_client()

    def _set_embedder(self):
        """Sets the document embedder based on the input embedding model or embedding source.
        If no embedding model is given, a default model is used.
        """
        if self.embedding_source.lower() == "openai":
            self.logger.info("Using OpenAI embedding model to embed the documents")
            return AzureOpenAIEmbeddings(
                deployment=self.embedding_model,
                chunk_size=1,
            )

        elif self.embedding_source.lower() == "huggingface":
            self.logger.info(
                "Using Hugging Face embedding model to embed the documents"
            )
            if self.embedding_model is None:
                self.embedding_model = DEFAULT_HUGGING_FACE_MODEL
            return HuggingFaceEmbeddings(model_name=self.embedding_model)

        else:
            raise NotImplementedError(
                f"Embedding source {self.embedding_source} not available. Only embedding models from 'HuggingFace' or 'OpenAI' are currently available."
            )

    def _create_azuresearch_index(self):
        client = SearchIndexClient(
            self.store_path, AzureKeyCredential(self.store_password)
        )
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(self.embedder.embed_query("Text")),
                vector_search_profile_name="myHnswProfile",
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ],
        )
        client.create_index(
            SearchIndex(name=self.store_id, fields=fields, vector_search=vector_search)
        )

    def _set_client(self):
        """Sets the vector store client"""
        if self.store_service.lower() == "chroma":
            return PersistentClient(db_path=Path(self.store_path))
        elif self.store_service.lower() == "azuresearch":
            return SearchClient(
                self.store_path,
                index_name=self.store_id,
                credential=AzureKeyCredential(self.store_password),
            )
        else:
            raise NotImplementedError(
                f"Vectore store {self.store_service} not available. Only 'chroma' or 'azuresearch' are currently available."
            )

    def _add_embedding_model_to_metadata(self, metadatas: list) -> List[dict]:
        """
        Add the embedding model used to embed the documents to the metadata
        """
        new_metadatas = []
        for metadata in metadatas:
            new_metadata = copy.deepcopy(metadata)
            new_metadata[dm.EMBEDDING_MODEL] = self.embedding_model
            new_metadatas.append(new_metadata)
        return new_metadatas

    def _set_langchain_client(self):
        """Set the vector store langchain client"""
        if self.store_service.lower() == "chroma":
            self.logger.info("Initializing ChromaDB")
            _ = self.client.get_or_create_collection(self.store_id)
            return Chroma(
                store_id=self.store_id,
                embedding_function=self.embedder,
                client=self.client,
            )
        elif self.store_service.lower() == "azuresearch":
            return AzureSearch(
                azure_search_endpoint=self.store_path,
                azure_search_key=self.store_password,
                index_name=self.store_id,
                embedding_function=self.embedder.embed_query,
            )
        else:
            raise NotImplementedError(
                f"Vectore store {self.store_service} not available. Only 'chroma' or 'azuresearch' are currently available."
            )

    def add_documents(self, chunked_documents: List[Document]) -> int:
        """
        Add new incoming chunked documents to the vector store
        If the collection/index specified by store_id is not empty, replace all content
        Add metadata regarding the embedding model
        """
        n_docs_added = 0
        if len(chunked_documents) > 0:
            n_docs_in_collection = 0
            if self.store_service.lower() == "chroma":
                n_docs_in_collection = self.client.get_or_create_collection(
                    self.store_id
                ).count()
            elif self.store_service.lower() == "azuresearch":
                n_docs_in_collection = self.client.get_document_count()

            if n_docs_in_collection > 0:
                self.logger.info(
                    f"Vector store already contains {n_docs_in_collection} documents. Replacing everything."
                )
                if self.store_service.lower() == "chroma":
                    self.client.delete_collection(self.store_id)
                    _ = self.client.get_or_create_collection(self.store_id)
                elif self.store_service.lower() == "azuresearch":
                    index_client = SearchIndexClient(
                        self.store_path, AzureKeyCredential(self.store_password)
                    )
                    index_client.delete_index(self.store_id)
                    self._create_azuresearch_index()

            n_docs_added = len(chunked_documents)
            self.logger.info(f"Adding {n_docs_added} new incoming chunked documents")
            self.langchain_client.add_documents(chunked_documents)

            return n_docs_added

    def count_documents(self):
        n_docs_in_collection = None
        if self.store_service.lower() == "chroma":
            n_docs_in_collection = self.client.get_or_create_collection(
                self.store_id
            ).count()
        elif self.store_service.lower() == "azuresearch":
            n_docs_in_collection = self.client.get_document_count()
        return n_docs_in_collection

    def get_documents(self):
        return self.langchain_client.get(include=["documents", "metadatas"])
