import json
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset

from src.constants import TIMESTAMP_FORMAT, MetricMetadata
from src.RAG.utils import initialize_logger, set_llm

from ragas import evaluate
from ragas.llms import LangchainLLM
from ragas.metrics import (AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness)
from ragas.metrics.base import MetricWithLLM

mm, rm, hm = MetricMetadata.RagasMetrics(), MetricMetadata.RagasMetadata(), MetricMetadata.HitRateMetadata()

class RagEvaluationMetrics:

    """
    Evaluation class that:
        1. Evaluates a dataset consisting of ground truth context, question and rag generated answer and top-k documents on (depends on input):
            1. Hitrate
            2. Context Precision
            3. Context Recall
            4. Faithfulness
            5. Answer Relevancy

    Input
    --------
    dataset:
        dict
        {
            question_id: uuid {
                question: str,
                context_id: uuid.int,
                answer: str,
                context: str,
                contexts: {uuid.int: str, ...}
            }, ...
        }
    temperature:
        int
        controls the level of randomness or "creativity" in the evaluation prompts. 
    --------
    output:
        pd.Dataframe
        Dataframe with evaluation scores per ground truth context, question and rag generated answer and retreived document pair on provided evaluation metrics
        Dict
        Dictionary with aggregated evaluation scores on inputted evaluation metrics
    """

    def __init__(
        self, data: dict, temperature: float = 0.2, output_folder="data\evaluation_runs"
    ):
        self.data = data
        self.logger = initialize_logger(self.__class__.__name__)
        self.llm = set_llm(temperature=temperature)
        self.ragas_model = LangchainLLM(llm=self.llm)
        self.ragas_metric_dict = {
            mm.CONTEXT_PRECISION: self.initialize_ragas_metric(ContextPrecision),
            mm.CONTEXT_RECALL: self.initialize_ragas_metric(ContextRecall), 
            mm.FAITHFULNESS: self.initialize_ragas_metric(Faithfulness), 
            mm.ANSWER_RELEVANCY: self.initialize_ragas_metric(AnswerRelevancy)
        }
        self.output_folder = output_folder
    
    def _to_hf_dataset(self) -> Dataset:
        """
        Transforming the JSON file into HuggingFace Dataset format for ragas evaluatuion
        """

        #get all keys from the subdicts in the dictionary to find the provided evaluation strategy ground_truth vs no ground truth
        eval_columns = list(self.data[next(iter(self.data))].keys())

        self.logger.info("transforming dataset to Huggingface format")

        transformed_dict = {
            rm.QUESTION: list(subdict[rm.QUESTION] for subdict in self.data.values()),
            rm.GENERATED_ANSWER: list(subdict[rm.GENERATED_ANSWER] for subdict in self.data.values()),
            rm.CONTEXTS: [list(subdict[rm.CONTEXTS].values()) for subdict in self.data.values()],
        }

        if rm.GROUND_TRUTH_ANSWER in eval_columns:
            transformed_dict[rm.GROUND_TRUTH_ANSWER] = [[subdict[rm.GROUND_TRUTH_ANSWER]] for subdict in self.data.values()]
    
        hf_dataset = Dataset.from_dict(transformed_dict)

        return hf_dataset
    
    def initialize_ragas_metric(self, metric: MetricWithLLM):
        """Initialize Ragas metric"""
        initialized_metric = metric(name=str(metric.__name__))
        initialized_metric.llm = self.ragas_model
        initialized_metric.init_model()

        return initialized_metric


    def _run_hit_rate_metric(self) -> pd.DataFrame:
        """
        Calculation of the Hit Rate Score
        """
        self.logger.info("calculating hit rate score")

        eval_results = [
            {
                hm.HIT_RATE: 1 if self.data[question_id]["context_id"] in self.data[question_id][rm.CONTEXTS].keys()
                else 0,
                hm.RETRIEVED: list(self.data[question_id][rm.CONTEXTS].keys()),
                hm.EXPECTED : self.data[question_id]["context_id"],
                "question_id": question_id,
            }
            for question_id in self.data.keys()
        ]

        hit_rate_df = pd.DataFrame(eval_results)

        return hit_rate_df
    


    def _run_ragas_metrics(self, metrics: List[MetricWithLLM], dataset: Dataset) -> pd.DataFrame:
        """
        Excecutes the RAGAS evaluation for the provided RAGAS metrics
        """
 
        result = evaluate(dataset=dataset, metrics=metrics)
        ragas_df = result.to_pandas()
        ragas_df["question_id"] = list(self.data.keys())

        return ragas_df   

    def _run_metrics(self, metrics: List) -> pd.DataFrame:
        """
        Runs the evaluation for the provided metrics
        """
        hf_dataset = self._to_hf_dataset()
        
        selected_ragas_metrics = [self.ragas_metric_dict[metric] for metric in metrics if metric in self.ragas_metric_dict.keys()]
        selected_hitrate_metric = [metric for metric in metrics if metric == "hitrate"]

        if selected_ragas_metrics and selected_hitrate_metric:
            return self._run_ragas_metrics(
                metrics=selected_ragas_metrics,
                dataset=hf_dataset,
            ).merge(self._run_hit_rate_metric(), on="question_id", how="inner")

        if selected_hitrate_metric:
            return self._run_hit_rate_metric()

        if selected_ragas_metrics:
            return self._run_ragas_metrics(
                metrics=selected_ragas_metrics,
                dataset=hf_dataset,
            )

    def _save(self, data: dict) -> None:
        """Saves dict to json."""

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

        output_file_name = os.path.join(self.output_folder, f"evaluation_data_{timestamp}.json")

        data = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in item.items()} for item in data]

        with open(output_file_name, "w") as f:
            json.dump(data, f, indent=4)
    
    def run(self, metrics:List[str], output_eval_dataset:bool) -> pd.DataFrame:
        """Runs the evaluation and returns a dataset or evaluation scores and saves to CSV or JSON"""

        eval_dataset = self._run_metrics(metrics=metrics)

        average_scores = {metric: eval_dataset[metric].mean() for metric in metrics}
        self.logger.info(average_scores)

        if output_eval_dataset:
            eval_dict = eval_dataset.to_dict('records')
            self._save(data=eval_dict)
        
        return eval_dataset  
    

if __name__ == "__main__":
    rag_evaluation_metrics = RagEvaluationMetrics(data={"test"}, temperature=0.2)