import pandas as pd
import re

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate

import warnings
warnings.filterwarnings("ignore")

# import constants
from src.constants import Neo4jMetadata, DEFAULT_QA_MODEL
NM = Neo4jMetadata()


class CypherQAProcessor:
    """
    A module that makes a connection to a graph database, then prompts an LLM to write a cypher query to
    answer a question using this graph datbase. The output of the cypher query is returned and the LLM also
    provides an answer based on this graph context.

    Input
    --------
    llm_model:
        str
        the OpenAi model used for the llm
    graph_url:
        str
        The url to the graph database. By default, bolt://localhost:7687, which applies to a graph database running locally in Neo4J
    graph_username:
        str
        username to the graph database, by default neo4j
    graph_password:
        str
        password of the graph database
    num_examples:
        The number of user questions and corresponding cypher examples to add to the prompt for few-shot learning
    examples_path:
        str
        path to the Excel file with columns 'Question' and 'Answer', containing example questions and corresponding 
        correctly formulated cypher queries
    kg_prompt_path:
        str
        The prompt with instructions aimed at the llm
    """

    def __init__(
        self,
        llm_model=DEFAULT_QA_MODEL,
        graph_url=NM.GRAPH_URL,
        graph_username=NM.GRAPH_USERNAME,
        graph_password=NM.GRAPH_PASSWORD,
        num_examples=4,
        examples_path=None,
        kg_prompt_path=None,
    ):
        # Initialize the AzureChatOpenAI instance
        self.llm = AzureChatOpenAI(deployment_name=llm_model, model_name=llm_model, temperature=0)

        # Initialize the Neo4jGraph instance
        self.graph = Neo4jGraph(url=graph_url, username=graph_username, password=graph_password)

        # Get some settings about the cypher prompt and cypher examples
        self.num_examples = num_examples
        if (self.num_examples > 0) and examples_path is not None:
            # Load the examples Excel file into a DataFrame
            examples_df = pd.read_excel(examples_path)
            # Convert DataFrame to a list of dictionaries
            self.examples_list = examples_df.to_dict(orient="records")

        self.kg_prompt_path = kg_prompt_path

    def _clean_cypher_examples(self, input_string):
        """
        Clean the output from the SemanticSimilarityExampleSelector,
        so that only the user questions and cypher queries are returned.

        Input
        --------
        input_string:
            str
            The example selector output string which should be cleaned
        """
        # Remove the introductory sentence
        cleaned_string = input_string.replace("Give the cypher code based on the user question\n\n", "")

        # Regular expression pattern to find and remove the last incomplete question (without cypher code after it)
        pattern = r"\n\n\s*([^\s].*)?\?\s*$"

        # Remove the last incomplete sentence ending in a question mark
        cleaned_string = re.sub(pattern, "", cleaned_string)

        return cleaned_string

    def _get_cypher_examples(self, user_question):
        """
        Select cypher examples using the langchain SemanticSimilarityExampleselector
        from a list of user questions and corresponding cypher examples.
        These could be used to enter into an llm prompt to improve the quality of cypher generation.

        Input
        --------
        user_question:
            str
            The user question for which we want to select the top k most
            similar examples from our list of example questions.

        """

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15",
        )

        example_prompt = PromptTemplate(
            input_variables=["Question", "Answer"],
            template="{Question}\n{Answer}",
        )

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            # The list of examples available to select from.
            self.examples_list,
            # The embedding class used to produce embeddings which are used to measure semantic similarity.
            embeddings,
            # The VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # The number of examples to produce.
            k=self.num_examples,
        )

        similar_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="Give the cypher code based on the user question",
            suffix="{question}\n",
            input_variables=["question"],
        )

        examples = similar_prompt.format(question=user_question)
        return self._clean_cypher_examples(examples)

    def _get_cypher_prompt(self, examples):
        """
        Create a prompt for cypher generation using the langchain
        PromptTemplate class and potentially add in examples

        We have to insert the examples directly into the text before
        creating the prompt template, because adding other options to
        the prompt is currently not supported by graphCypherQAChain

        Input
        --------
        examples:
            str
            A string with cypher generation examples that should be inserted
            into the prompt
        """

        with open(self.kg_prompt_path, "rt", encoding="utf-8") as f:
            system_message_template = f.read()

        CYPHER_TEMPLATE = system_message_template.format(examples=examples)

        CYPHER_GENERATION_PROMPT = PromptTemplate(input_variables=["schema", "question"], template=CYPHER_TEMPLATE)

        return CYPHER_GENERATION_PROMPT

    def run_KG_qa(self, Q, verbose=False):
        """
        run a question answering LLM

        Input
        --------
        Q:
            str
            The user question to be answered
        verbose:
            bool
            Whether the cypher output and intermediate results
            should be printed.
        """
        if self.num_examples > 0:
            cypher_examples = self._get_cypher_examples(user_question=Q)
        else:
            cypher_examples = ""

        CYPHER_GENERATION_PROMPT = self._get_cypher_prompt(cypher_examples)

        chain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            validate_cypher=True,
            verbose=verbose,  # To see what is happening inside the chain
            return_intermediate_steps=True,
        )

        result = chain(Q)

        return result
