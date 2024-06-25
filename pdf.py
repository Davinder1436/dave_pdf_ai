import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: 
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000,
        chunk_overlap  = 1000,
        length_function = len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embeddings)
    vector_store.save_local("fiass_index")

def get_conversational_chain():
    prompt_template = """
  Anwser the question from provided context as detailed as possible, don't provide wrong answer

  Context =\n {context}\n
  Question = \n {question}\n

  Answer:
 """
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(template= prompt_template , input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(question_input):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("fiass_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question_input)

    chain = get_conversational_chain()

    response = chain({"input_documents":docs,"question":question_input},return_only_outputs=True)

    st.write("Question: ",question_input)

    st.write("Reply: ",response["output_text"])

def main():
    st.set_page_config("chat with multiple pdfs")
    st.header("Chat with multiple pdfs")

    user_question = st.text_input("Ask your question")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("menu")
        pdf_docs = st.file_uploader("upload your pdf",type=["pdf"],accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("done")
        
if __name__ == "__main__":
    main()



    