from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that converts English to Spanish. Please respond to the input the user gives and translate it into Spanish."),
    ("user", "Question: {question}")
])

st.title('LangchainXGemini Sample')
input_text = st.text_input("Enter the text you want to translate:")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
