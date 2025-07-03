import pytest
from pathlib import Path
from src.eval.synthetic_data_generator import ContextQuestionDatasetGenerator
from src.RAG.vector_store import ChromaVectorStore
from src.RAG.utils import set_openai_env_vars
from pathlib import Path

PATH_TO_OPENAI_CONFIG = Path(__file__).parent.parent.joinpath("config.json")

embedding_source = "HuggingFace"
db_path = Path.cwd().joinpath("data", f"chromaDB_test_eval")
chroma_db = ChromaVectorStore(
    chunked_documents=[], embedding_source=embedding_source, db_path=db_path
)

set_openai_env_vars(PATH_TO_OPENAI_CONFIG)


@pytest.fixture
def data_generator():
    return ContextQuestionDatasetGenerator(load_from="vectorstore", chroma_db=chroma_db)


def test_generate_dataset(data_generator):
    dataset = data_generator.generate(dataset_size=2, temperature=0.8)
    assert isinstance(dataset, dict)
    assert len(dataset["questions"]) == 2
