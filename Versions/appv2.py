# Import necessary libraries
import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Title of the Streamlit app
st.title("LangChain-Powered Data Analysis Bot")

# Step 1: File uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

# Display basic data analysis functionalities if a CSV is uploaded
if uploaded_file is not None:
    # Step 2: Read and display dataset using Pandas
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())  # Display the first few rows of the dataset

    # Display basic info about the dataset
    st.write("### Basic Information about the Dataset")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Step 3: Create simple query options
    st.write("### Perform Simple Queries on Dataset")

    # Let user select a column to display
    if st.checkbox("Show specific column"):
        selected_column = st.selectbox("Select a column", df.columns)
        st.write(f"Displaying data from column: {selected_column}")
        st.dataframe(df[selected_column])

    # Show basic statistics
    if st.checkbox("Show dataset statistics"):
        st.write("### Dataset Statistics")
        st.write(df.describe())

    # Button to display full dataset if needed
    if st.button("Show Full Dataset"):
        st.write(df)
    
    # ---- LangChain Chatbot Section ----
    st.write("## Ask Questions about the Dataset")
    
    # Create a string representation of the dataset (but now, include entire dataset)
    data_sample = df.to_string()  # Pass the entire dataset as a string to the model (not limited to 10 rows)

    # Prompt Template - include the dataset context
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a data analysis assistant. Here's the dataset:\n{data_sample}"),
        ("user", "Question: {question}")
    ])

    # Streamlit input for user question
    input_text = st.text_input("Ask a question about your dataset:")

    # Process the input text
    if input_text:
        # Initialize the LLM model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        output_parser = StrOutputParser()

        # Combine the prompt, LLM, and output parser
        chain = prompt | llm | output_parser

        # Special handling for the 'average' query
        if 'average' in input_text.lower():
            # Try to extract the column name based on user's input
            # We'll look for the column name based on the dataset and query context
            selected_column = None
            for col in df.columns:
                if col.lower() in input_text.lower():
                    selected_column = col
                    break
            
            if selected_column:
                # Calculate the average of the specified column
                try:
                    avg_value = df[selected_column].mean()
                    st.write(f"The average value for the column '{selected_column}' is: {avg_value}")
                except Exception as e:
                    st.write(f"Error calculating average for column '{selected_column}': {e}")
            else:
                st.write("Could not find the column to calculate the average. Please specify a valid column.")
        else:
            # For other queries, use LangChain LLM
            st.write("### Response from AI:")
            st.write(chain.invoke({'question': input_text}))
else:
    st.write("Please upload a CSV file to get started.")
