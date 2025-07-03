# Import modules
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain

# Import own modules that are needed
from src.RAG.qa_retriever import CustomQARetriever
from src.KG.kg_qa_retriever import CypherQAProcessor

# import constants
from src.constants import Neo4jMetadata, DEFAULT_QA_MODEL
NM = Neo4jMetadata()


class IntegratedQAProcessor:
    """
    A module that combines the context from a QA LLM based on a graph database and a QA LLM
    based on a vector store of documents into a new LLM which aims to answer a user question.

    Input
    --------
    vector_store_retriever:
        ChromaVectorStore
        Vector store with documents to be used as context for the qa
    combine_documents_strategy:
        str
        the strategy used to combine the candidate chunked documents that are given to the LLM
    llm_model:
        str
        the OpenAi model used for the llm
    num_examples:
        The number of cypher examples to add to the cypher qa prompt
        If no examples are available, then set it to 0
    examples_path:
        str
        path to the Excel file containing user questions and cypher examples.
    graph_password:
        str
        password of the graph database
    vectordb_prompt_path:
        str
        path to the prompt for the vector database llm
    kg_prompt_path:
        str
        path to the prompt with instructions aimed at the cypher llm
    integrated_prompt_path:
            str
        The prompt with instructions aimed at the cypher llm

    """

    def __init__(
        self,
        vector_store_retriever,
        combine_documents_strategy="stuff_with_citation",
        llm_model=DEFAULT_QA_MODEL,
        num_examples=4,
        examples_path=None,
        graph_password=NM.GRAPH_PASSWORD,
        vectordb_prompt_path=None,
        kg_prompt_path=None,
        integrated_prompt_path=None,
    ):
        # Initialize OpenAI params
        self.llm = AzureChatOpenAI(deployment_name=llm_model, model_name=llm_model, temperature=0)
        self.integrated_prompt_path = integrated_prompt_path
        prompt = self._load_prompt(path_to_prompt = integrated_prompt_path)

        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        # ------------
        # Initialize the documentstore LLM

        self.qa_retriever = CustomQARetriever(
            retriever=vector_store_retriever,
            prompt_path=vectordb_prompt_path,
            combine_documents_strategy=combine_documents_strategy,
        )

        # ------------
        # also initialize the graph QA LLM
        self.cypher_qa_processor = CypherQAProcessor(
            graph_password=graph_password,
            num_examples=num_examples,
            examples_path=examples_path,
            kg_prompt_path=kg_prompt_path,
        )

    def _load_prompt(self, path_to_prompt = None):
        """Load a prompt template for the integrated retriever"""

        with open(path_to_prompt, "rt", encoding="utf-8") as f:
            system_message_template = f.read()

        integration_prompt = PromptTemplate(input_variables=["schema", "question"], template=system_message_template)

        return integration_prompt

    def process_question(self, Q, verbose=False, return_context=False):
        """
        Run a question answering LLM which uses context from documents and from
        a graph database.

        Input
        --------
        Q:
            str
            The user question to be answered
        verbose:
            bool
            Whether the context and individual answers of KG and document Q&A should be printed
        return_context:
            bool
            Whether the function should return the KG and document context along with the answer in a dictionary.
        """
        # ------ Get the context from the KG Q&A LLM ------

        try:
            KG_result = self.cypher_qa_processor.run_KG_qa(Q, verbose=False)
            KG_answer = KG_result["result"]
            KG_context = KG_result["intermediate_steps"][1]["context"]

            if verbose:
                print(f"KG context: {KG_context}")
                print(f"KG answer: {KG_answer}")
                print("\n-----------------------------\n")

        except Exception as e:
            print("An incorrect cypher query was produced")
            KG_context = "[]"

        # ------ Get the context from the vector store Q&A LLM ------

        # create LLM question answering instance
        LLM_result = self.qa_retriever.qa(Q)
        LLM_context = [doc.page_content.replace("\n", " ") for doc in LLM_result["source_documents"]]

        if verbose:
            print(f"Document context: {LLM_context}")
            print(f"Document answer: {LLM_result}")
            print("\n-----------------------------\n")

        # ------ Run the chain with the given question ------
        result = self.llm_chain.run(question=Q, documents=LLM_context, graph_context=KG_context)

        if return_context:
            return {
                "result": result,
                "source_documents": LLM_result["source_documents"],
                "graph_context": KG_context,
            }

        else:
            return result

    def qa(self, Q):
        # process question for evaluation
        return self.process_question(Q, return_context=True)
