import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_pdf_text(pdf_docs):
    text = ""
    # goes through all pdfs and pages and gets all texts and returns
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_spliter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_spliter.split_text(text)
    return chunks


# using OpenAI embeddings and then storing using FAISS(could be swapped for pinecone)
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your docs: ")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and Click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store with embeddings
                vectorstore = get_vectorstore(text_chunks)


if __name__ == "__main__":
    main()
