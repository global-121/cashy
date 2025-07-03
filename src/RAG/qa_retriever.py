import json
from pathlib import Path
from typing import Any, Optional

from langchain.callbacks.manager import Callbacks
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.vectorstores.base import VectorStoreRetriever

from src.constants import DocumentMetadata, DEFAULT_QA_MODEL
from src.RAG.utils import initialize_logger

dm = DocumentMetadata()


class CustomQARetriever:
    """
    A module that retrieves candidate chunked documents from the database given a question
    Then prompts an LLM with the question and the candidate chunked documents to answer the question.
    Optionally reads in custom Json QA prompt template from path

    Input
    --------
    retriever:
        VectorStoreRetriever
        retriever initialized from chromaDB vectorstore
        this retriever embeds the query using the same embedding function as used to initialize the chromaDB
    path_to_prompt_messages:
        Path
        path to the json file containing the prompt messages
    return_source_documents:
        bool
        whether to include the source documents retrieved from the vector store in the LLM result
    combine_document_strategy:
        str
        the strategy used to combine the candidate chunked documents that are given to the LLM
    temperature:
        float
        the sampling temperature
    citation_metadata:
        str
        the metadata the LLM should receive and use for the source citation in the answer
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        prompt_path: Path = None,
        return_source_documents: bool = True,
        combine_documents_strategy: str = "stuff_with_citation",
        temperature: float = 0.1,
        citation_metadata: str = dm.CELL_INDEX,
        qa_model: str = DEFAULT_QA_MODEL,
    ):
        self.logger = initialize_logger(self.__class__.__name__)
        self.retriever = retriever
        self.default_prompt_path = Path(__file__).parent.joinpath(
            "prompts", "prompt_messages_citation.json"
        )
        self.prompt = self._load_prompt_if_not_none(prompt_path)
        self.return_source_documents = return_source_documents
        self.combine_documents_strategy = combine_documents_strategy
        self.temperature = temperature
        self.citation_metadata = citation_metadata
        self.qa_model = qa_model
        self.qa = self._set_retrieval_qa()

    def _load_prompt_if_not_none(self, path_to_prompt_messages: Path) -> PromptTemplate:
        """
        Load a list of messages from a json path that contains messages for the chat prompt template
        """
        if not path_to_prompt_messages:
            self.logger.info("No prompt path provided, using default prompt.")
            path_to_prompt_messages = self.default_prompt_path

        with open(path_to_prompt_messages, "rt", encoding="utf-8") as f:
            prompt_messages = json.load(f)

        prompt_messages = [(actor, message) for actor, message in prompt_messages]

        return ChatPromptTemplate.from_messages(prompt_messages)

    def _set_retrieval_qa(self) -> RetrievalQA:
        if self.combine_documents_strategy == "stuff":
            self.logger.info(
                "Setting up the QA retriever with 'stuff' as document combining strategy"
            )
            return RetrievalQA.from_llm(
                llm=AzureChatOpenAI(
                    deployment_name=self.qa_model,
                    model_name=self.qa_model,
                    temperature=self.temperature,
                ),
                prompt=self.prompt,
                retriever=self.retriever,
                return_source_documents=self.return_source_documents,
            )
        elif self.combine_documents_strategy == "stuff_with_citation":
            self.logger.info(
                "Setting up the QA retriever with 'stuff_with_citation' as document combining strategy. Metadata will be included for stuffed documents to use for citation."
            )
            return RetrievalQAMetadata.from_llm(
                llm=AzureChatOpenAI(
                    deployment_name=self.qa_model,
                    model_name=self.qa_model,
                    temperature=self.temperature,
                ),
                citation_metadata=self.citation_metadata,
                prompt=self.prompt,
                retriever=self.retriever,
                return_source_documents=self.return_source_documents,
            )
        else:
            raise NotImplementedError(
                f"Strategy '{self.combine_documents_strategy}' not available for combining documents, currently only 'stuff' and 'stuff_with_citation' is implemented."
            )


class RetrievalQAMetadata(RetrievalQA):
    """
    A Langchain RetrievalQA class child
    Includes metadata when stuffing the retrieved documents into the LLM QA prompt
    """

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        citation_metadata: str,
        prompt: Optional[PromptTemplate] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt, callbacks=callbacks)
        template = f"Context:\nDocument[{{{citation_metadata}}}]: {{page_content}}"
        document_prompt = PromptTemplate(
            input_variables=["page_content", citation_metadata], template=template
        )

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=callbacks,
        )

        return cls(
            combine_documents_chain=combine_documents_chain,
            callbacks=callbacks,
            **kwargs,
        )
