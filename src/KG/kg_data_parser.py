import pandas as pd
import os
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL
from urllib.parse import quote



class KG_DataParser:
    """
    This module takes Excel files containing nodes and relationships as input and
    parses them to a knowledge graph in .ttl file format, such that it can be
    put into an application like Neo4J.

    Input
    --------
    nodes_data:
        str
        the path to the Excel file with node information
    relationships_data:
        str
        The path to the Excel file with relationship information
    ontology_data:
        str
        The path to the Excel file containing ontological information such as constraints
    ttl_file:
        str
        the path where the output ttl file will be created
    namespace:
        str
        the namespace to use when creating the triples in turtle format
    """

    def __init__(
        self,
        nodes_data=None,
        relationships_data=None,
        ontology_data=None,
        ttl_file=None,
        namespace="http://www.redcross.org/510/",
    ):
        # initializing variables that will be used in the rest of the class
        self.ttl_file = ttl_file

        self.nodes_df = pd.read_csv(str(nodes_data), sep=";")
        self.rel_df = pd.read_csv(str(relationships_data), sep=";")
        self.ontology_df = pd.read_csv(ontology_data, sep=";")

        self.RedCross = Namespace(namespace)

        self.distinct_node_types = self.ontology_df["Subject"][self.ontology_df["Object"] == "Class"].unique()

        # Create an empty RDF graph
        self.graph = Graph()

    def __create_uri(self, name: str) -> URIRef:
        """Take a string and return a valid URI (Uniform Resource Identifier). This means that
        spaces should be replaced by _ characters"""
        quoted = quote(name.replace(" ", "_"))

        return self.RedCross[quoted]

    def add_nodes(self):
        """Add nodes and rdf type relationship to the graph"""

        for index, row in self.nodes_df.iterrows():
            # Create a unique URI for each entity:
            # If it is an instance, we add the index for uniqueness
            entity_uri = self.__create_uri(row["Name"] + "/" + str(index))
            # For the class names we do not add any index
            node_type_uri = self.__create_uri(row["Node_type"])

            # Add triple for entity type
            self.graph.add((entity_uri, RDF.type, node_type_uri))

            # Add labels to node
            self.graph.add((entity_uri, RDFS.label, Literal(row["Name"])))
            # self.graph.add((entity_uri, RDFS.label, Literal(row['Node_type'])))

            # Iterate over the properties and add triples for non-blank values
            for column in self.nodes_df.columns:
                if column not in ["Node_type", "Name"] and pd.notnull(row[column]):
                    # Assuming properties are in the red cross namespace:
                    property_uri = self.__create_uri(column)
                    property_value = Literal(row[column])  # not sure if it should always be literal
                    self.graph.add((entity_uri, property_uri, property_value))

    def add_ontology(self):
        """Add ontology information to the graph"""

        for index, row in self.ontology_df.iterrows():
            print(f"{row['Subject']} - {row['Property']} - {row['Object']}")

            subject = self.__create_uri(row["Subject"])

            if row["Object"] == "AsymmetricProperty":
                obj = OWL.AsymmetricProperty

            elif row["Object"] == "SymmetricProperty":
                obj = OWL.SymmetricProperty
            elif row["Object"] == "Class":
                obj = OWL.Class
            elif row["Object"] == "TransitiveProperty":
                obj = OWL.TransitiveProperty
            else:
                obj = self.__create_uri(row["Object"])

            if row["Property"] == "rdf:type":
                self.graph.add((subject, RDF.type, obj))
            elif row["Property"] == "rdfs:subClassOf":
                self.graph.add((subject, RDFS.subClassOf, obj))
            elif row["Property"] == "rdf:domain":
                self.graph.add((subject, RDFS.domain, obj))
            elif row["Property"] == "rdf:range":
                self.graph.add((subject, RDFS.range, obj))
            elif row["Property"] == "owl:inverseOf":
                self.graph.add((subject, OWL.inverseOf, obj))
            elif row["Property"] == "owl:equivalentClass":
                self.graph.add((subject, OWL.equivalentClass, obj))
            else:
                property_uri = self.__create_uri(row["Property"])
                self.graph.add((subject, property_uri, obj))

        for node_type in self.distinct_node_types:
            print(node_type)
            node_type_uri = self.__create_uri(node_type)
            self.graph.add((node_type_uri, RDFS.label, Literal(node_type)))

    def add_relationships(self):
        """Add relationships to the graph that are specified from an input file"""

        # Iterate over rel_df to create relationships
        for index, row in self.rel_df.iterrows():
            print(f"{row['Node_1']} - {row['Relationship']} - {row['Node_2']}")

            if row["Node_1"] in self.distinct_node_types:
                # If the entity in the relationship is a class (node type),
                # then we do not need to add any index
                entity1_uri = self.__create_uri(row["Node_1"])
            else:
                # If the entity is an instance, then we need to match it to the unique uri with index
                # which we assigned to that entity
                entity1_uri = self.__create_uri(
                    self.nodes_df.loc[self.nodes_df["Name"] == row["Node_1"], "Name"].values[0]
                    + "/"
                    + str(self.nodes_df.loc[self.nodes_df["Name"] == row["Node_1"]].index[0])
                )

            if row["Node_2"] in self.distinct_node_types:
                # If the entity in the relationship is a class (node type),
                # then we do not need to add any index
                entity2_uri = self.__create_uri(row["Node_2"])

            else:
                # If the entity is an instance, then we need to match it to the unique uri with index
                # which we assigned to that entity
                entity2_uri = self.__create_uri(
                    self.nodes_df.loc[self.nodes_df["Name"] == row["Node_2"], "Name"].values[0]
                    + "/"
                    + str(self.nodes_df.loc[self.nodes_df["Name"] == row["Node_2"]].index[0])
                )

            # If the relationship is 'a', it's an RDF type relationship
            # in that case we want to look at node types and not at names
            if row["Relationship"] == "a":
                self.graph.add((entity1_uri, RDF.type, entity2_uri))

            else:
                # Assuming relationships are in the Red Cross namespace
                relationship_uri = self.__create_uri(row["Relationship"])
                self.graph.add((entity1_uri, relationship_uri, entity2_uri))

    def write_graph(self):
        """Write graph to turtle format, which can be loaded in to Neo4j"""

        # Serialize the RDF graph to Turtle format
        turtle_data = self.graph.serialize(format="turtle")

        # Write the Turtle data to a file
        directory = os.path.dirname(self.ttl_file)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.ttl_file, "w") as file:
            file.write(turtle_data)
