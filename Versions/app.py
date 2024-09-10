import streamlit as st
import pandas as pd

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

else:
    st.write("Please upload a CSV file to get started.")
