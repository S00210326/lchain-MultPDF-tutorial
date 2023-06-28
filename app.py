import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import pinecone
from htmlTemplates import css, bot_template, user_template
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


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


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


# using OpenAI embeddings and then storing using FAISS(could be swapped for pinecone)
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_name = "langchain1"

    vectorstore = Pinecone.from_texts(
        texts=text_chunks, embedding=embeddings, index_name=index_name
    )

    return vectorstore


# when user submits a question
def handle_userinput(user_question):
    if st.session_state.conversation is not None:  # Add this check
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(
                    user_template.replace("{{MSG}}", message.content),
                    unsafe_allow_html=True,
                )
            else:
                st.write(
                    bot_template.replace("{{MSG}}", message.content),
                    unsafe_allow_html=True,
                )
    else:
        st.write("Please process the PDFs first before asking a question.")


def main():
    load_dotenv()
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV,  # next to api key in console
    )
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    # allows to use css from the imported template
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your docs: ")
    if user_question:
        handle_userinput(user_question)
    # replacing personalized template, loading straight html code

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

                # create conversation chain
                # this allows you to get history of conversation and returns you the next element
                # Needs to be persistant(Especially for using STREAMLIT where it re-initialises alot)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
