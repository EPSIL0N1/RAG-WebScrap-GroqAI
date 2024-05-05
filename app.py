import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    print("Loading...1")
    st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text')
    st.session_state.loader = WebBaseLoader("https://python-adv-web-apps.readthedocs.io/en/latest/scraping.html")
    st.session_state.docs = st.session_state.loader.load()
    print("Loading...2")
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap = 200

    )
    
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    print("Loading...3")
    st.session_state.vectors = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    print("Loading...4")

st.title("Chat Groq")
llm = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = "llama3-70b-8192"
)

prompt = ChatPromptTemplate.from_template(
    
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}    

"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here!")

if prompt:
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])
    
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant Chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------------")
