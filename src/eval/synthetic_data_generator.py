import random
from uuid import uuid4

from langchain.prompts import PromptTemplate
from langchain.schema import Document

from src.RAG.utils import initialize_logger, set_llm, uuid_hash

from typing import List


class ContextQuestionDatasetGenerator:

    """
    Context Question Generator class that:
        1. sets up a connection to an AzureOpenAI gpt-35-turbo LLM
        2. Uses LLM and prompt to creates {dataset_size} number of context, question pairs.
        3. Saves context, question pairs as JSON.

    Input
    --------
    dataset_size:
        int
        The number of context, question pairs generated in the dataset.
    temperature:
        int
        controls the level of randomness or "creativity" in the generated text.
    chunk_docs:
        List[Documents]
        Documents on which the synthetic datast is generated
    """

    def __init__(self, temperature: float = 0.8):
        self.logger = initialize_logger(self.__class__.__name__)
        self.llm = set_llm(temperature=temperature)

    def _generate_qa_pairs(
        self, chunked_documents: list[Document], llm, dataset_size
    ) -> dict:
        """
        Generating the questions
        """
        QuestionGenerationPrompt = PromptTemplate.from_template(
            """
            Context information is below.

            ---------------------
            {context}
            ---------------------

            Given the context information and no prior knowledge.
    
            You are a Teacher/ Professor. Your task is to setup 
            one question which can be anwered with the context information provided. 
            Restrict the question to the context information provided."
            """
        )

        self.logger.info(f"Starting dataset generation with {dataset_size} questions")

        dataset = {}
        for _ in range(dataset_size):
            document = random.choice(chunked_documents)
            context_id = document.metadata["document_id"]
            context = document.page_content
            prompt = QuestionGenerationPrompt.format(context=context)
            question = llm.predict(prompt)
            question_id = uuid_hash(question)
            dataset[question_id] = {
                "question": question,
                "context_id": context_id,
                "context": context,
            }

        self.logger.info("Dataset generation completed")
        return dataset

    def generate(
        self, chunked_documents: List[Document], dataset_size: int
    ) -> dict:
        """
        Generating the questions
        """

        dataset = self._generate_qa_pairs(
            chunked_documents=chunked_documents, llm=self.llm, dataset_size=dataset_size
        )

        return dataset
