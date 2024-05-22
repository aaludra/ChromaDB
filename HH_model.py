import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

# Set up the Hugging Face model
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
hf_pipeline = pipeline(
    "text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100
)

# Directory where the articles are stored
ARTICLES_DIR = "new_articles"

# Load documents
loader = DirectoryLoader(ARTICLES_DIR, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize the embedding and vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = "db"
vectordb = Chroma.from_documents(
    documents=texts, embedding=embedding, persist_directory=persist_directory
)
retriever = vectordb.as_retriever()

# Use Hugging Face pipeline with LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


def process_llm_response(llm_response):
    result = llm_response["result"]
    sources = [doc.metadata["source"] for doc in llm_response["source_documents"]]
    return result, sources


def query_documents(query):
    llm_response = qa_chain(query)
    result, sources = process_llm_response(llm_response)
    return result, sources


# Streamlit UI
st.title("Document Query System")

query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        result, sources = query_documents(query)
        st.write("Response:", result)
        st.write("Sources:")
        for source in sources:
            st.write(source)
