import streamlit as st


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your docs: ")

    with st.sidebar:
        st.subheader("Your Documents")


if __name__ == "__main__":
    main()
