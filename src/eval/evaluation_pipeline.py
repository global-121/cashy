from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from langchain.schema import Document
from typing import List, Optional

from src.eval.rag_evaluation_metrics import RagEvaluationMetrics
from src.eval.synthetic_data_generator import ContextQuestionDatasetGenerator
from src.constants import MetricMetadata, DocumentMetadata as dm
from src.RAG.qa_retriever import CustomQARetriever
from src.RAG.utils import uuid_hash, initialize_logger
from src.RAG.vector_store import ChromaVectorStore

mm, rm = (MetricMetadata(), MetricMetadata.RagasMetadata())


@dataclass
class EvaluationArguments:
    """
    Arguments for the RAG (Retrieval-Augmented Generation) Evaluation pipeline.
    """

    qa_retriever: CustomQARetriever = field(
        metadata={
            "help": "Retriever for extracting relevant information from documents."
        }
    )
    vector_store: Optional[ChromaVectorStore] = field(
        metadata={"help": "Vector store for retrieving relevant documents."}
    )

    dataset_size: Optional[int] = field(
        metadata={"help": "Number of context, question pairs to be generated."},
        default=10,
    )

    generative_temperature: Optional[int] = field(
        metadata={
            "help": "Controls the level of randomness or 'creativity' in generating the questions of the synthetic dataset."
        },
        default=0.8,
    )
    evaluative_temperature: Optional[int] = field(
        metadata={
            "help": "Controls the level of randomness or 'creativity' in calculating the evaluation scores. Default should be 0.2 according to RAGAS."
        },
        default=0.2,
    )
    output_eval_dataset: Optional[bool] = field(
        metadata={
            "help": "If True, a dataset is returned and saved, if False a score dictionary is returned and saved."
        },
        default=True,
    )
    ground_truth_filepath: Optional[Path] = field(
        metadata={
            "help": "The file path to the ground_truth json file with question, answer pairs."
        },
        default=None,
    )


class EvaluationPipeline:

    """
    EvaluationPipeline class that runs the evaluation flow consisting of:
        1. Generating a synthetic context, question dataset
        2. Running the RAG module on the questions in the dataset
        3. Evaluation the answers, and retrieved context for generating the answer with the Evaluation Module

    Input
    --------
    qa_retriever:
        CustomQARetriever
        Initialized instance of the RAG module custom QAretriever
    vector_store:
        ChromaVectorStore
        Initialized instance of the RAG module ChromaDB Vector store. Needed for synthetic context, question generation.

    Output
    --------
    evaluation_results
        pd.Dataframe
        returned as variable when .evaluate method is called
        JSON
        saved to evaluation_runs folder
    """

    def __init__(
        self,
        qa_retriever: CustomQARetriever,
        vector_store: ChromaVectorStore = None,
        output_folder: str = "data\evaluation_runs"
    ):
        self.logger = initialize_logger(self.__class__.__name__)
        self.qa_retriever = qa_retriever
        if vector_store:
            self.documents = self._load_from_vector_store(vector_store)
        else:
            self.documents = None
        self.default_ground_truth_template = Path(__file__).parent.joinpath(
            "ground_truth_template", "ground_truth_eval_template.csv"
        )
        self.output_folder = output_folder

    def _load_from_vector_store(
        self, vector_store: ChromaVectorStore
    ) -> List[Document]:
        """
        Loads a List of Documents from a ChromaDB vectorstore and transforms it into the Langchain Document format
        """
        documents = vector_store.get_documents()

        docs_langchain_format = [
            Document(page_content=document, metadata={"document_id": id})
            for id, document in zip(documents["ids"], documents["documents"])
        ]

        if not docs_langchain_format:
            raise ValueError(f"No documents found in the vector store")

        return docs_langchain_format

    def _run_rag(self, questions: dict) -> dict:
        """
        Queries the RAG module with the questions from the generated synthetic dataset


        Input
        --------
        data:
            dict
            {
                question_id: {
                    question: str
                }, ...
            }

        Output
        --------
        data:
            dict
            {
                question_id: {
                    question: str,
                    answer: str,
                    contexts: List[uuid.int, str]}, ..
            }
        """

        for question_id in questions.keys():
            (
                questions[question_id][rm.GENERATED_ANSWER],
                questions[question_id][rm.CONTEXTS],
            ) = (
                {},
                {},
            )

            result = self.qa_retriever.qa(questions[question_id][rm.QUESTION])

            context = {
                f"{doc.metadata['document_id']}.{doc.metadata['nth_chunk']}": doc.page_content
                for doc in result["source_documents"]
            }
            
            questions[question_id][rm.GENERATED_ANSWER] = result["result"]
            questions[question_id][rm.CONTEXTS] = context

        return questions

    def _format_ground_truth_dataset(self, ground_truth_filepath: str):
        """Transforms a csv into json structure to fit evaluation pipeline"""
        df = pd.read_csv(ground_truth_filepath, sep=";")

        df["hash"] = df["question"].apply(uuid_hash)

        df = df.set_index("hash")

        gt_dict = df.to_dict(orient="index")

        return gt_dict

    def _validate_metrics(self, metrics: List, eval_type: str):
        """
        Validates if the metrics are supported for the evaluation method chosen
        """

        if eval_type == "ground_truth":
            for metric in metrics:
                if metric not in mm.ALLOWED_EVALS_GROUND_TRUTH:
                    raise ValueError(
                        f"Metric {metric} is not a valid metric '{eval_type}', please list one of {mm.ALLOWED_EVALS_GROUND_TRUTH}."
                    )
        elif eval_type == "synthetic":
            for metric in metrics:
                if metric not in mm.ALLOWED_EVALS_SYNTHETHIC:
                    raise ValueError(
                        f"Metric {metric} is not a valid metric for evaluation method '{eval_type}', please list one of {mm.ALLOWED_EVALS_SYNTHETHIC}."
                    )
        else:
            raise ValueError(
                f"Only evaluation type 'synthetic' or 'ground_truth' are supported, not {eval_type}"
            )

    def _evaluate(self, data, metrics, evaluative_temperature, output_eval_dataset):
        retrieved_data = self._run_rag(questions=data)

        rag_evaluation_metrics = RagEvaluationMetrics(
            data=retrieved_data, temperature=evaluative_temperature, output_folder=self.output_folder
        )

        eval_data = rag_evaluation_metrics.run(
            metrics=metrics,
            output_eval_dataset=output_eval_dataset,
        )

        return eval_data

    def evaluate_ground_truth(
        self,
        metrics_list: List,
        ground_truth_filepath: Path = None,
        evaluative_temperature: float = 0.2,
        output_eval_dataset: bool = True,
    ):
        """
        Evaluates a ground truth dataset on a metric_list

        input
        -------------
        metrics_list:
            list
            the evaluation metrics on which the RAG setup is evaluated
        ground_truth_filepath:
            str
            The file path to the ground_truth json file with question, answer pairs
        evaluative_temperature:
            float
            controls the level of randomness or "creativity" in calculating the evaluation scores. Default should be 0.2 according to RAGAS
        output_eval_dataset:
            Bool
            if True, a dataset is returned and saved, if False a score dictionary is returned and saved.
        ------------
        output
        eval_data:
            pd.Dataframe | Dictionary
            based on output_eval_dataset either a dataframe with evaluation metrics on each pair in the dataset is returned or a dictionary with aggregrate evaluation metrics
        """

        self._validate_metrics(metrics=metrics_list, eval_type="ground_truth")

        if not ground_truth_filepath:
            ground_truth_filepath = self.default_ground_truth_template
            self.logger.info(
                f"No filepath to ground truth template given, using default ground truth template in {self.default_ground_truth_template}"
            )

        data = self._format_ground_truth_dataset(
            ground_truth_filepath=ground_truth_filepath
        )

        eval_data = self._evaluate(
            data, metrics_list, evaluative_temperature, output_eval_dataset
        )

        return eval_data

    def evaluate_synthetic(
        self,
        metrics_list: List,
        dataset_size: int,
        generative_temperature: float = 0.8,
        evaluative_temperature: float = 0.2,
        output_eval_dataset: bool = True,
    ):
        """ "
        Evaluates a synthetic generated dataset on a metric_list

        input
        -------------
        metrics_list:
            list
            the evaluation metrics on which the RAG setup is evaluated
        dataset_size:
            int
            number of context, question pairs to be generated
        generative_temperature:
            float
            controls the level of randomness or "creativity" in generating the questions of the synthetic dataset.
        evaluative_temperature:
            float
            controls the level of randomness or "creativity" in calculating the evaluation scores. Default should be 0.2 according to RAGAS
        output_eval_dataset:
            Bool
            if True, a dataset is returned and saved, if False a score dictionary is returned and saved.
        ------------
        output
        eval_data:
            pd.Dataframe | Dictionary
            based on output_eval_dataset either a dataframe with evaluation metrics on each pair in the dataset is returned or a dictionary with aggregrate evaluation metrics
        """

        self._validate_metrics(metrics=metrics_list, eval_type="synthetic")

        generator = ContextQuestionDatasetGenerator(temperature=generative_temperature)

        data = generator.generate(
            chunked_documents=self.documents,
            dataset_size=dataset_size,
        )

        eval_data = self._evaluate(
            data, metrics_list, evaluative_temperature, output_eval_dataset
        )

        return eval_data

