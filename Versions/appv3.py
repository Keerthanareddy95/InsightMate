import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="pastel")

st.title("Insightmate")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Basic Information about the Dataset")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("### Dataset Summary")
    st.write(df.describe())

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    st.write("### Auto-Generated Dashboard")

    if len(numeric_columns) > 0:
        st.write("#### Distribution of Numerical Columns")
        for i in range(0, len(numeric_columns), 2):
            cols = st.columns(2)
            for idx, column in enumerate(numeric_columns[i:i + 2]):
                with cols[idx]:
                    plt.figure(figsize=(6, 4))
                    sns.histplot(df[column], kde=True, color='lightblue', bins=30)
                    plt.title(f'Distribution of {column}')
                    st.pyplot(plt)

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

    if len(numeric_columns) > 1:
        st.write("#### Correlation Heatmap of Numerical Columns")
        plt.figure(figsize=(8, 5))
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title('Correlation Heatmap')
        st.pyplot(plt)

    if df.isnull().sum().sum() > 0:
        st.write("#### Missing Values Heatmap")
        plt.figure(figsize=(8, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title('Missing Values in the Dataset')
        st.pyplot(plt)

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

    st.write("### Generate Custom Visualizations")

    plot_type = st.selectbox("Select Plot Type", ["Histogram", "Bar Plot", "Line Plot", "Scatter Plot", "Pie Chart", "Box Plot", "Correlation Heatmap", "Violin Plot", "Pair Plot"])

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
        if st.button("Generate Correlation Heatmap"):
            plt.figure(figsize=(8, 5))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title('Correlation Heatmap')
            st.pyplot(plt)

    elif plot_type == "Violin Plot":
        selected_column = st.selectbox("Select column for violin plot", numeric_columns)
        if st.button("Generate Violin Plot"):
            plt.figure(figsize=(8, 4))
            sns.violinplot(data=df[selected_column], palette="Set3")
            plt.title(f'Violin Plot of {selected_column}')
            st.pyplot(plt)

    elif plot_type == "Pair Plot":
        selected_columns = st.multiselect("Select columns for pair plot (up to 3)", numeric_columns, default=numeric_columns[:3])
        if st.button("Generate Pair Plot"):
            if len(selected_columns) > 1:
                sns.pairplot(df[selected_columns], diag_kind="kde", palette="coolwarm")
