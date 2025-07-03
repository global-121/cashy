from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KG_DataParserArguments:
    """
    Arguments pertaining to how the Chroma Vectorstore should be instantiated.
    """

    nodes_data: Optional[str] = field(
        default=None,
        metadata={"help": "the path to the Excel file with node information"},
    )

    relationships_data: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the Excel file with relationship information"},
    )

    ontology_data: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the Excel file containing ontological information such as constraints"},
    )

    ttl_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path where the output ttl file will be created"},
    )

    namespace: Optional[str] = field(
        default="http://www.redcross.org/510/",
        metadata={"help": " The namespace to use when creating the triples in turtle format"},
    )


@dataclass
class CypherQAProcessorArguments:
    """
    Arguments pertaining to the CypherQAProcessor.
    """

    kg_prompt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a text file containing the prompt messages"},
    )

    examples_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to an Excel with columns 'Question' and 'Answer', containing example questions and corresponding correctly formulated cypher queries"
        },
    )

    num_examples: Optional[int] = field(
        default=4,
        metadata={"help": "The number of cypher examples to use in the cypherQA prompt"},
    )


@dataclass
class IntegratedQAProcessorArguments:
    """
    Arguments pertaining to how the Chroma Vectorstore should be instantiated.
    """

    integrated_prompt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a text file containing the prompt message for the integrated qa processor"},
    )
