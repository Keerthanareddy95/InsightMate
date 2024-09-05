# Import necessary libraries
import streamlit as st
import pandas as pd

# Title of the Streamlit app
st.title("LangChain-Powered Data Analysis Bot")

# Step 1: File uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

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

else:
    st.write("Please upload a CSV file to get started.")
