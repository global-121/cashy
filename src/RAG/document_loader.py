import copy
from collections import OrderedDict
from typing import List
from langchain.schema import Document
import pandas as pd
from src.constants import DocumentMetadata
from src.RAG.utils import initialize_logger, uuid_hash
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders.sitemap import SitemapLoader

dm = DocumentMetadata()


class DocumentLoader:
    """
    Document loading class that:
        1. Loads documents from path
        2. Uses the hash of the page content to create a unique documentID
        3. Returns list of Langchain Document

    Input
    --------
    document_path:
        str
        Location of the raw document(s)
    document_id:
        str
        Unique identifier of the document
    document_type:
        str
        Data type of document(s)
    kwargs:
        dict
        Input parameters that are conditional to the document type

    Output
    --------
    The output is a list in Document type. Each element has two attributes:
    page_content:
        str
        Content of documents
    metadata:
        dictionary
        Information of each page_content
    """

    output: List[Document]

    def __init__(
        self,
        document_type: str,
        document_path: str = None,
        document_id: str = None,
        **kwargs: dict,
    ):
        self.logger = initialize_logger(self.__class__.__name__)
        self.document_path = document_path
        self.document_id = document_id
        self.document_type = document_type
        self.__dict__.update(kwargs)
        self.loader = self._set_loader()

    def _set_loader(self):
        """
        Instantiates a document loader based on the document type. PDF and Excel are allowed to be loaded
        PDF document is loaded from the BaseLoader langchain class
        Excel document is loaded from a custom loder inherented from the langchain class
        """
        if self.document_type.lower() == "googlesheet":
            self.logger.info("Loading from Google Sheet with pandas.")
            sheet_name = "Q%26As"
            url = f"https://docs.google.com/spreadsheets/d/{self.document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            return url
        elif self.document_type.lower() == "manual.121.global":
            self.logger.info("Loading from manual.121.global.")
            return SitemapLoader(
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
        else:
            raise NotImplementedError(
                f"Loader of document type {self.document_type} not available. Only loading of 'googlesheet' is currently implemented."
            )

    def _load(self):
        if self.document_type.lower() == "googlesheet":
            df = pd.read_csv(self.loader)
            # clean text
            df = df.rename(
                columns={"The Answer (can be multi-line)\n#ANSWER": "answer"}
            )
            df = df.rename(
                columns={"The Question (should be 1 line)\n#QUESTION": "question"}
            )
            df["text"] = df["question"] + " " + df["answer"]
            df["text"] = df["text"].str.replace(r"<[^<]+?>", "", regex=True)
            df = df.dropna(subset=["text"])
            df["text"] = df["text"].astype(str)
            # extra filters
            df = df[df["Visible?\n#VISIBLE"] == "Show"]  # only visible entries
            df = df.rename(
                columns={
                    [c for c in df.columns if "#CATEGORY" in c][0]: "category",
                    [c for c in df.columns if "#SUBCATEGORY" in c][0]: "subcategory",
                    [c for c in df.columns if "#SLUG" in c][0]: "slug",
                    [c for c in df.columns if "#PARENT" in c][0]: "parent",
                }
            )
            df["category"] = df["category"].astype(int)
            df["subcategory"] = df["subcategory"].astype(int)
            df["cell_index"] = (
                df["category"].astype(str)
                + ";"
                + df["subcategory"].astype(str)
                + ";"
                + df["slug"].astype(str)
                + ";"
                + df["parent"].astype(str)
            )
            df = df[["text", "category", "subcategory", "slug", "parent", "cell_index"]]
            # map to langchain doc
            documents = DataFrameLoader(df, page_content_column="text").load()
        elif self.document_type.lower() == "manual.121.global":
            self.logger.info("Loading from manual.121.global.")
            documents = self.loader.load()
            # map to langchain doc
            documents = [
                Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc in documents
            ]
        else:
            raise NotImplementedError(
                f"Loader of document type {self.document_type} not available. Only loading of 'googlesheet' is currently implemented."
            )
        return documents

    def _check_emptiness(self, page_content: str) -> bool:
        """
        Check if document content is empty of text, by evaluating whether there are at least 3 consecutive alphabet characters in there
        """
        min_consecutive_chars = 3
        for letter in page_content:
            if letter.isalpha():
                min_consecutive_chars -= 1
            else:
                min_consecutive_chars = 3
            if min_consecutive_chars == 0:
                return False
        return True

    def _validate_loading(self, documents: List[Document]):
        """
        Validates if the text from documents is loaded
        Logs all documents from which no text has been loaded
        Raises info if one or more documents is not loaded
        Return a new list without documents which didn't pass emptiness check
        """
        sources_not_loaded = []
        valid_documents = []
        for doc in documents:
            if self._check_emptiness(doc.page_content):
                self.logger.warning(
                    f"Was not able to extract text from document {doc.metadata}; removed from the outputs."
                )
                sources_not_loaded.append(doc.metadata[dm.SOURCE])
            else:
                valid_documents.append(doc)

        sources_not_loaded = list(OrderedDict.fromkeys(sources_not_loaded))
        if sources_not_loaded:
            self.logger.info(
                f"Empty documents rendered from the following sources: {sources_not_loaded}"
            )

        return valid_documents

    def _add_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Adds an unique document ID to the document metadata
        Filters out duplicate documents
        """
        seen_hashes = []
        docs_with_metadata = []

        for doc in documents:
            new_metadata = copy.deepcopy(doc.metadata)
            page_content_hash = uuid_hash(content=doc.page_content)

            # Ensure that only docs with an unique documentID, which is the hash of the content, are kept
            if page_content_hash not in seen_hashes:
                new_metadata[dm.DOCUMENT_ID] = page_content_hash
                doc_with_new_metadata = Document(
                    page_content=doc.page_content, metadata=new_metadata
                )
                docs_with_metadata.append(doc_with_new_metadata)

            seen_hashes.append(page_content_hash)

        self.logger.info(
            f"Filtered out {len(seen_hashes) - len(docs_with_metadata)} incoming documents that were duplicate"
        )

        return docs_with_metadata

    def load(self) -> List[Document]:
        """
        Loads the documents
        Validates whether they are properly loaded
        Adds metadata
        Removes duplicate documents
        """
        self.logger.info(f"Started loading")
        documents = self._load()
        valid_documents = self._validate_loading(documents)
        valid_documents = self._add_metadata(valid_documents)
        self.logger.info(f"Loaded {len(valid_documents)} documents")
        return valid_documents
