# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Set a theme for plots using Seaborn
sns.set_theme(style="whitegrid", palette="pastel")

# Initialize the LangChain model
chat_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# prompt template for the model
prompt_template = ChatPromptTemplate.from_template("You are an assistant knowledgeable in data analysis. Answer the following question based on the dataset: {question}")

# Title of the Streamlit app
st.title("InsightMate")

# Step 1: File uploader for user to upload CSV files
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Step 2: Load the dataset using Pandas
    df = pd.read_csv(uploaded_file)

    # Step 3: Data cleaning check
    st.write("### Checking for Missing or Inconsistent Data")

    # Checking for missing values
    if df.isnull().values.any():
        st.warning("Your dataset contains missing values. Please clean the dataset and re-upload.")
        st.write(df.isnull().sum())  # Show columns with missing values
    else:
        st.success("No missing values detected. Proceeding with analysis.")

        # Display a preview of the dataset
        st.write("### Dataset Preview")
        st.dataframe(df.head()) 

        # Display basic dataset information
        st.write("### Basic Information about the Dataset")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.write("### Dataset Summary")
        st.write(df.describe())  # Show statistical summary

        # Identify numerical and categorical columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        # Step 4: Auto-Generated Dashboard with Key Visualizations
        st.write("### Auto-Generated Dashboard")

        # 4.1 Distribution plots for numerical columns
        if len(numeric_columns) > 0:
            st.write("#### Distribution of Numerical Columns")
            for i in range(0, len(numeric_columns), 2):  # Show 2 columns in one row
                cols = st.columns(2)  # Create 2 columns side-by-side
                for idx, column in enumerate(numeric_columns[i:i + 2]):
                    with cols[idx]:  # Place the plots in separate columns
                        plt.figure(figsize=(6, 4))  # Make the plot smaller
                        sns.histplot(df[column], kde=True, color='lightblue', bins=30)
                        plt.title(f'Distribution of {column}')
                        st.pyplot(plt)

        # 4.2 Bar plots for categorical columns
        if len(categorical_columns) > 0:
            st.write("#### Distribution of Categorical Columns")
            for i in range(0, len(categorical_columns), 2):
                cols = st.columns(2)
                for idx, column in enumerate(categorical_columns[i:i + 2]):
                    with cols[idx]:
                        top_categories = df[column].value_counts().nlargest(10)
                        filtered_df = df[df[column].isin(top_categories.index)]
                        plt.figure(figsize=(6, 4))
                        sns.countplot(x=column, data=filtered_df, palette="Set2", order=top_categories.index)
                        plt.title(f'Top 10 Categories in {column}')
                        plt.xticks(rotation=45)
                        st.pyplot(plt)
        
        #4.3 Pie charts for categorical columns
        if len(categorical_columns) > 0:
            st.write("#### Pie Charts for Categorical Columns")
            for i in range(0, len(categorical_columns), 2):
                cols = st.columns(2)
                for idx, column in enumerate(categorical_columns[i:i + 2]):
                    with cols[idx]:
                        pie_data = df[column].value_counts().nlargest(5)  
                        plt.figure(figsize=(6, 6))
                        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
                        plt.title(f'Pie Chart of {column}')
                        st.pyplot(plt)

        # 4.4 Correlation Heatmap for numerical columns
        if len(numeric_columns) > 1:
            st.write("#### Correlation Heatmap of Numerical Columns")
            plt.figure(figsize=(8, 5))
            correlation_matrix = df[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title('Correlation Heatmap')
            st.pyplot(plt)

        # Step 5: Query-based Dataset Filtering with Dropdown
        st.write("### Filter the Dataset by Unique Values in a Column")

        # Dropdown to select a column for filtering
        selected_filter_column = st.selectbox("Select a column to filter by unique values", df.columns)

        # Display unique values for the selected column
        unique_values = df[selected_filter_column].unique()

        # Dropdown to select one of the unique values for filtering
        selected_value = st.selectbox(f"Select a unique value from the '{selected_filter_column}' column", unique_values)

        # Filter the dataset based on the selected value
        filtered_df = df[df[selected_filter_column] == selected_value]
        st.write(f"### Filtered Dataset where `{selected_filter_column} == {selected_value}`")
        st.dataframe(filtered_df)

        # Step 6: Column selection for dataset exploration
        selected_columns = st.multiselect("Select columns for detailed exploration", df.columns)

        num_rows = st.slider("Select number of rows to display", min_value=5, max_value=100, value=10, step=5)

        if selected_columns:
            st.write(f"### Detailed view of selected columns (showing top {num_rows} rows)")
            st.dataframe(df[selected_columns].head(num_rows))  # Show the specified number of rows


        # Step 7: Custom visualizations
        st.write("### Generate Custom Visualizations")

        plot_type = st.selectbox("Select Plot Type", ["Histogram", "Bar Plot", "Line Plot", "Scatter Plot", "Pie Chart", "Box Plot", "Correlation Heatmap", "Violin Plot", "Pair Plot"])

        # depending on the plot type
        if plot_type == "Histogram":
            selected_column = st.selectbox("Select column for histogram", numeric_columns)
            if st.button("Generate Histogram"):
                plt.figure(figsize=(8, 4))
                sns.histplot(df[selected_column], bins=30, kde=True, color='lightblue')
                plt.title(f'Histogram of {selected_column}')
                st.pyplot(plt)

        elif plot_type == "Bar Plot":
            selected_column = st.selectbox("Select column for bar plot", df.columns)
            if st.button("Generate Bar Plot"):
                plt.figure(figsize=(8, 4))
                sns.countplot(x=selected_column, data=df, palette="Set2")
                plt.title(f'Bar Plot of {selected_column}')
                st.pyplot(plt)

        elif plot_type == "Line Plot":
            x_column = st.selectbox("Select X-axis column for line plot", df.columns)
            y_column = st.selectbox("Select Y-axis column for line plot", df.columns)
            if st.button("Generate Line Plot"):
                plt.figure(figsize=(8, 4))
                sns.lineplot(x=df[x_column], y=df[y_column], color='coral')
                plt.title(f'Line Plot: {x_column} vs {y_column}')
                st.pyplot(plt)

        elif plot_type == "Scatter Plot":
            x_column = st.selectbox("Select X-axis column", df.columns)
            y_column = st.selectbox("Select Y-axis column", df.columns)
            if st.button("Generate Scatter Plot"):
                plt.figure(figsize=(8, 4))
                sns.scatterplot(x=df[x_column], y=df[y_column], hue=df[y_column], palette="coolwarm")
                plt.title(f'Scatter Plot: {x_column} vs {y_column}')
                st.pyplot(plt)

        elif plot_type == "Pie Chart":
            selected_column = st.selectbox("Select column for pie chart", categorical_columns)
            if st.button("Generate Pie Chart"):
                pie_data = df[selected_column].value_counts()
                plt.figure(figsize=(6, 6))
                plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
                plt.title(f'Pie Chart of {selected_column}')
                st.pyplot(plt)

        elif plot_type == "Box Plot":
            selected_column = st.selectbox("Select column for box plot", numeric_columns)
            if st.button("Generate Box Plot"):
                plt.figure(figsize=(8, 4))
                sns.boxplot(data=df[selected_column], palette="muted")
                plt.title(f'Box Plot of {selected_column}')
                st.pyplot(plt)

        elif plot_type == "Correlation Heatmap":
            if len(numeric_columns) > 1:
                plt.figure(figsize=(8, 5))
                correlation_matrix = df[numeric_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
                plt.title('Correlation Heatmap')
                st.pyplot(plt)


        elif plot_type == "Violin Plot":
            selected_column = st.selectbox("Select column for violin plot", numeric_columns)
            if st.button("Generate Violin Plot"):
                plt.figure(figsize=(8, 4))
                sns.violinplot(data=df[selected_column], palette="pastel")
                plt.title(f'Violin Plot of {selected_column}')
                st.pyplot(plt)

        elif plot_type == "Pair Plot":
            selected_columns = st.multiselect("Select columns for pair plot", df.columns)
            if len(selected_columns) >= 2:
                sns.pairplot(df[selected_columns], palette="husl")
                st.pyplot(plt)

        # Step 8: User queries for insights via Gemini
        st.write("### Ask Questions to Gemini for Informative Insights")
        user_question = st.text_input("Enter your question about the dataset:")

        # Convert the dataset to JSON for passing it to Gemini
        dataset_json = df.to_json(orient='split') 

        max_rows = 100  
        limited_df = df.head(max_rows)
        limited_dataset_json = limited_df.to_json(orient='split')

        if st.button("Get Answer"):
            if user_question:
                # Include a portion of the dataset in the prompt
                prompt = f"Dataset (in JSON format): {limited_dataset_json}. \nQuestion: {user_question}"

                # Generate response from the LLM
                response = chat_llm.invoke(prompt)

                # Access the content of the response
                formatted_response = response.content.replace('\n', '\n\n')

                st.write("### Gemini's Answer")
                st.write(formatted_response)
            else:
                st.warning("Please enter a question.")

else:
    st.info("Please upload a dataset to begin.")