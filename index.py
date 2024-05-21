import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = (
    "sk-proj-yWQaJQjO5jzP0kuLc4CQT3BlbkFJ1RhMlTGe5DmCOKdn0Mnz"
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
embedding = OpenAIEmbeddings()
persist_directory = "db"
vectordb = Chroma.from_documents(
    documents=texts, embedding=embedding, persist_directory=persist_directory
)
vectordb.persist()
vectordb = None
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever()

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True
)


def process_llm_response(llm_response):
    result = llm_response["result"]
    sources = [doc.metadata["source"] for doc in llm_response["source_documents"]]
    return result, sources


# Streamlit UI
st.title("Document Retrieval and QA System")
query = st.text_input("Enter your query:")
if st.button("Submit"):
    if query:
        llm_response = qa_chain(query)
        result, sources = process_llm_response(llm_response)
        st.write("### Response")
        st.write(result)
        st.write("### Sources")
        for source in sources:
            st.write(source)
    else:
        st.write("Please enter a query.")
