from dotenv import load_dotenv
from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
from langchain.chains import GraphQAChain
from langchain.graphs.networkx_graph import NetworkxEntityGraph, KnowledgeTriple
from langchain.graphs import NetworkxEntityGraph
import tiktoken
from langchain.chat_models import ChatOpenAI
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


def chunk_text_by_tokens(text, max_tokens=3000):
    """
    Splits the text into chunks of a specified number of tokens.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


with open("ALAllints2.txt") as f:
    all_text = f.read()
openai_llm = OpenAI(temperature=0)
index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
# Split the text into chunks
chunks = chunk_text_by_tokens(all_text)
# Initialize an empty graph
G = nx.DiGraph()

# Process each chunk
for chunk in chunks:
    # Create a graph index for the chunk
    chunk_graph = index_creator.from_text(chunk)
    triples = chunk_graph.get_triples()

    # Add triples to your DiGraph
    for triple in triples:
        G.add_edges_from([(triple[0], triple[1], {"relation": triple[2]})])


entity_graph = NetworkxEntityGraph(G)
triples = entity_graph.get_triples()

print(triples)
chain = GraphQAChain.from_llm(openai_llm, graph=entity_graph, verbose=True)
question = "Who owns alstom and what is it ? "

print(chain.run(question))


# Plot the graph
plt.figure(figsize=(8, 5), dpi=500)
pos = nx.spring_layout(G, k=3, seed=0)

nx.draw_networkx_nodes(G, pos, node_size=150)
nx.draw_networkx_edges(G, pos, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=3)
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4)

# Display the plot
plt.axis("off")
plt.show()
