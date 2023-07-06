# import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.memory import ConversationBufferMemory
from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, Docx2txtLoader
import networkx as nx
import matplotlib.pyplot as plt

# from langchain.chains import ConversationalRetrievalChain, GraphQAChain
# from langchain.chat_models import ChatOpenAI
# import pinecone
from htmlTemplates import css, bot_template, user_template
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

# def get_pdf_text(pdf_doc):
#     text = ""
#     # Reads all pages of the PDF and gets all text
#     pdf_reader = PdfReader(pdf_doc)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_spliter = CharacterTextSplitter(
#         separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
#     )
#     chunks = text_spliter.split_text(text)
#     return chunks


# pdf_text = get_pdf_text("ALAllints2.pdf")

# # text = get_text_chunks(pdf_text)

# index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
# graph = index_creator.from_text(pdf_text)
# graph.get_triples()


index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))

document_loader = Docx2txtLoader(document_path="ALAllints2.docx")

document = document_loader.load_document()
index_creator.add_document(document)

graph = index_creator.graph

graph.get_triples()


# Create graph
G = nx.DiGraph()
G.add_edges_from(
    (source, target, {"relation": relation})
    for source, relation, target in graph.get_triples()
)

# Plot the graph
plt.figure(figsize=(8, 5), dpi=300)
pos = nx.spring_layout(G, k=3, seed=0)

nx.draw_networkx_nodes(G, pos, node_size=2000)
nx.draw_networkx_edges(G, pos, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=12)
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Display the plot
plt.axis("off")
plt.show()
