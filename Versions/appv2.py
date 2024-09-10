import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.title("InsightMate")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Basic Information about the Dataset")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    st.write("### Perform Simple Queries on Dataset")

    if st.checkbox("Show specific column"):
        selected_column = st.selectbox("Select a column", df.columns)
        st.write(f"Displaying data from column: {selected_column}")
        st.dataframe(df[selected_column])

    if st.checkbox("Show dataset statistics"):
        st.write("### Dataset Statistics")
        st.write(df.describe())

    if st.button("Show Full Dataset"):
        st.write(df)
    
    st.write("## Ask Questions about the Dataset")
    
    data_sample = df.to_string()

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a data analysis assistant. Here's the dataset:\n{data_sample}"),
        ("user", "Question: {question}")
    ])

    input_text = st.text_input("Ask a question about your dataset:")

    if input_text:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        output_parser = StrOutputParser()

        chain = prompt | llm | output_parser

        if 'average' in input_text.lower():
            selected_column = None
            for col in df.columns:
                if col.lower() in input_text.lower():
                    selected_column = col
                    break
            
            if selected_column:
                try:
                    avg_value = df[selected_column].mean()
                    st.write(f"The average value for the column '{selected_column}' is: {avg_value}")
                except Exception as e:
                    st.write(f"Error calculating average for column '{selected_column}': {e}")
            else:
                st.write("Could not find the column to calculate the average. Please specify a valid column.")
        else:
            st.write("### Response from AI:")
            st.write(chain.invoke({'question': input_text}))
else:
    st.write("Please upload a CSV file to get started.")
