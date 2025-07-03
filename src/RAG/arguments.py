
from dataclasses import dataclass, field
from typing import Optional
from typing import List
from src.constants import DocumentMetadata
from pathlib import Path


dm = DocumentMetadata()

@dataclass
class DocumentLoaderArguments:
    """
    Arguments pertaining to how the documents should be loaded.
    """

    document_path: str = field(
        metadata={
            "help": "location of the raw document(s)"
        }
    )
    document_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "type of document(s) to load, f.e. PDFDirectory"
        },
    )
    load_from_existing_vector_db: Optional[str] = field(
        default=False,
        metadata={
            "help": "Whether to load, chunk, embed documents and create a vectorDB or to read an existing vectorDB from path"
        },
    )


@dataclass
class ExcelLoaderArguments(DocumentLoaderArguments):
    """
    Arguments pertaining to how an Excel document should be loaded
    """

    sheet_name: Optional[str] = field(
        default="Sheet1",
        metadata={
            "help": "The name of the Excel sheet to load"
        }
    )
    column_names: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Names of the Excel columns to load"
        },
    )



@dataclass
class DocumentChunkingArguments:
    """
    Arguments for the document chunking class
    """

    chunk_size: int = field(
        metadata={
            "help": "Size of the chunks to attempt to create, actual sizes might be larger or smaller due to keeping sentences intact"
        }
    )

    chunk_overlap: int = field(
        metadata={
            "help": "Overlap between the chunks (stride in sliding window)"
        }
    )

    chunking_strategy: Optional[str] = field(
        default="tokenizedsentencesplitting",
        metadata={
            "help": "The strategy to use for chunking the documents"
        }
    )

    separator: Optional[str] = field(
        default=" ",
        metadata={
            "help": "The separator to insert between sentences within a chunk"
        },
    )
    pipeline: Optional[str] = field(
        default="en_core_web_sm",
        metadata={
            "help": "The model to use for splitting the documents into sentences"
        },
    )
    pipeline: Optional[str] = field(
        default="cl100k_base",
        metadata={
            "help": "The encoding for tokenized chunking. 'cl100k_base' is appropriate for gpt3.5"
        },
    )



@dataclass
class ChromaVectorStoreArguments:
    """
    Arguments pertaining to how the Chroma Vectorstore should be instantiated.
    """

    db_path: Path = field(
        metadata={
            "help": "path to store or to load the Chroma database from"
        }
    )
    embedding_source: str = field(
        metadata={
            "help": "Source for the model used for embedding the documents"
        }
    )
    embedding_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the embedding model, if None it reverts to a default embedding model depending on the source"
        },
    )
    collection_name: Optional[str] = field(
        default="chunked_document_embeddings",
        metadata={
            "help": "Name of the ChromaDB collection used to store the documents and embeddings"
        },
    )


@dataclass
class QARetrievalArguments:
    """
    Arguments pertaining to how the Chroma Vectorstore should be instantiated.
    """

    prompt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the json file containing the prompt messages"},
    )
    return_source_documents: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to include the source documents retrieved from the vector store in the LLM result"
        },
    )
    combine_document_strategy: Optional[str] = field(
        default="stuff_with_citation",
        metadata={
            "help": "The strategy used to combine the candidate chunked documents that are given to the LLM. Stuff with citation makes sure that the LLM receives citation metadata to cite in the answer"
        },
    )
    temperature: Optional[float] = field(
        default=0.1,
        metadata={"help": "The sampling temperature"},
    )

    citation_metadata: Optional[str] = field(
        default=dm.SOURCE,
        metadata={"help": "The metadata the LLM should receive and use for the source citation in the answer"},
    )

    similarity_search: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to filter the retrieved documents from the vectorDB by a similarity score threshold"},
    )

    similarity_score_threshold: Optional[float] = field(
        default=0.2,
        metadata={"help": "The similarity score threshold used for filtering the retrieved documents"},
    )
       
    top_k_candidate_documents: Optional[int] = field(
        default=3,
        metadata={
            "help": "Maximum amount of candidate documents to retrieve for a specific question"
        },
    )
    
    qa_model: Optional[str] = field(
        default="510-chat",
        metadata={
            "help": "Te model to use for question answering"
        },
    )

