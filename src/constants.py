import logging
from pathlib import Path
from typing import List

from typing_extensions import TypedDict, NotRequired

LOGGING_LEVEL = logging.INFO
TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M%S"
DEFAULT_QA_MODEL = "gpt-35-turbo"

class DocumentMetadata:
    """Document class metadata keys"""

    __slots__ = ()
    DOCUMENT_ID = "document_id"
    SOURCE = "source"
    PAGE = "page"
    PAGE_NAME = "page_name"
    PAGE_NUMBER = "page_number"
    COLUMN_NAME = "column_name"
    CELL_INDEX = "cell_index"
    NTH_CHUNK = "nth_chunk"
    EMBEDDING_MODEL = "embedding_model"
    URLS = "urls"

class Neo4jMetadata:

    """configuration of connection to Neo4J graph database"""

    __slots__ = ()

    GRAPH_URL ="bolt://localhost:7687"
    GRAPH_USERNAME = "neo4j"
    GRAPH_PASSWORD = "redcross510"

class AzureChatMetadata:

    """Azure deployment configurations"""

    __slots__ = ()

    DEPLOYMENT_NAME = "gpt-35-turbo"
    MODEL_NAME = "gpt-35-turbo"
    OPENAI_API_VERSION = "2023-05-15"  # This AzureOpenAI api version needs to be after May 2023 (approximately) for Azure to find the resource.


class MetricMetadata:
    """Metric configurations"""

    __slots__ = ()

    class RagasMetrics:
        # RAGAS metrics

        __slots__ = ()

        CONTEXT_RECALL = "ContextRecall"
        CONTEXT_PRECISION = "ContextPrecision"
        FAITHFULNESS = "Faithfulness"
        ANSWER_RELEVANCY = "AnswerRelevancy"

    class RagasMetadata:
        # RAGAS metrics

        __slots__ = ()

        # HuggingFace Dataset formatting for RAGAS
        QUESTION = "question"
        GROUND_TRUTH_ANSWER = "ground_truths"
        GENERATED_ANSWER = "answer"
        CONTEXTS = "contexts"
        GROUND_TRUTH_CONTEXT = "ground_truth_context"

    class HitRateMetadata:
        __slots__ = ()

        HIT_RATE = "hitrate"
        RETRIEVED = "retrieved"
        EXPECTED = "expected"

    ALLOWED_EVALS_SYNTHETHIC = [
        RagasMetrics.CONTEXT_PRECISION,
        RagasMetrics.FAITHFULNESS,
        RagasMetrics.ANSWER_RELEVANCY,
        HitRateMetadata.HIT_RATE,
    ]
    ALLOWED_EVALS_GROUND_TRUTH = [
        RagasMetrics.CONTEXT_RECALL,
        RagasMetrics.CONTEXT_PRECISION,
        RagasMetrics.FAITHFULNESS,
        RagasMetrics.ANSWER_RELEVANCY,
    ]
