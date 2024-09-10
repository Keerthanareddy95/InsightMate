# 🌟 InsightMate: Your Personal Data Analysis Assistant 

**InsightMate** - an intuitive data analysis tool that helps you explore, visualize, and gain insights from your CSV datasets. With interactive visualizations and the integration of **Google's Gemini**, InsightMate provides instant answers to your dataset-related questions. This tool is designed to make data exploration easy and insightful! 🚀

## 🎯 Goal
The goal of this project is to create a user-friendly application that automates the process of dataset exploration and provides on-demand insights using AI, all through an interactive and visually rich dashboard.

## ✨ Key Features
- **Upload CSV Data**: Start by uploading any CSV file and get a preview of your dataset.
- **Data Cleaning Check**: Automatically checks for missing or inconsistent data.
- **Auto-Generated Dashboard**: Instantly generate distribution plots, bar plots, pie charts, correlation heatmaps, and more.
- **Custom Visualizations**: Create custom visualizations like histograms, scatter plots, line plots, and more using simple dropdowns on the columns of your preference.
- **Dataset Filtering**: Filter data by any column and view detailed exploration of selected columns.
- **Ask AI, Questions**: Interact with **Google’s Gemini** to ask questions about your dataset and get detailed insights.
  
## ⚙️ How to Run the Project

### 1. Clone the Repository
Start by cloning this repository to your local machine:

```bash
git clone https://github.com/Keerthanareddy95/InsightMate.git
cd InsightMate
```
## 2. Set Up Environment Variables
Create a `.env` file in the root of the project and add your API keys for Google Gemini and Langsmith:

```makefile
GOOGLE_API_KEY="your_google_api_key"
LANGCHAIN_API_KEY="your_langchain_api_key"
LANGCHAIN_PROJECT="your_langchain_project_name"
```
## 3. Install required libraries
Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```
## 4. Run the Application
Now, run the Streamlit app:

```bash
streamlit run final_app.py
```
Open the URL provided by Streamlit in your browser, and you're ready to explore! 🌍

## 📚 LLMOps with Langsmith

This project integrates **LLMOps** using **Langsmith** for tracing and monitoring AI model performance. By enabling Langsmith tracing, we can:

- **Track AI performance**: Understand how the model is responding to user queries.
- **Monitor latency**: Measure the time it takes for the model to process and return answers.
- **Ensure traceability**: Debug and audit model performance by tracking each invocation and its response.
  
![Screenshot 2024-09-10 132020](https://github.com/user-attachments/assets/c83aae59-12c5-406a-aeb2-e2caf0d8a5d5)

### Why Langsmith? 🤔

As AI models get more complex, managing their behavior, debugging, and improving efficiency becomes critical. **Langsmith** helps us manage this effectively by:

- Providing **detailed performance metrics** on LLM invocations.
- Allowing us to **track and tune model outputs** over time, ensuring our application delivers accurate insights.

## 🛠️ Development of InsightMate

- **Streamlit**: Used to build the web app interface and provide an interactive experience.
- **Pandas**: For loading and manipulating the dataset, and providing data summaries.
- **Matplotlib & Seaborn**: To generate various plots and visualizations of the dataset.
- **Google Gemini AI (via LangChain)**: To provide on-the-fly responses to user queries about their dataset.
- **Langsmith**: To monitor and trace the performance of AI responses and ensure quality results.

![Screenshot 2024-09-10 130541](https://github.com/user-attachments/assets/c559e736-317d-4af3-9beb-9faa97998590)


![Screenshot 2024-09-10 130604](https://github.com/user-attachments/assets/6b8da432-56ea-47a2-80c6-3cc8479c1d05)


![Screenshot 2024-09-10 130653](https://github.com/user-attachments/assets/71b39784-aebf-498d-8b05-3b2c8d263cb0)


![Screenshot 2024-09-10 130733](https://github.com/user-attachments/assets/e3eebd98-c93b-4acc-b182-46fc67012822)


![Screenshot 2024-09-10 130752](https://github.com/user-attachments/assets/a3a5e053-d61b-4417-8264-200638adc32c)


![Screenshot 2024-09-10 130809](https://github.com/user-attachments/assets/af6bda5d-d21b-412e-9443-12ff27e4dcff)


![Screenshot 2024-09-10 130916](https://github.com/user-attachments/assets/e53115e0-6ac7-4a36-9b67-ad98168af816)


![Screenshot 2024-09-10 131003](https://github.com/user-attachments/assets/a38e3319-94f7-4e4c-a723-45e5ab8f159f)


![Screenshot 2024-09-10 131143](https://github.com/user-attachments/assets/facc622c-d81e-4f3a-94ef-db9b568d8dae)


![Screenshot 2024-09-10 131606](https://github.com/user-attachments/assets/2934b838-b86d-4e38-959e-425741d24d19)


![Screenshot 2024-09-10 131642](https://github.com/user-attachments/assets/5d79492d-fc4c-4e99-8b3b-ebe8ff8e320b)

## 🔥 Future Enhancements

- Adding support for **multiple file formats** like Excel and JSON.
- Improve AI's ability to understand and analyze more complex datasets.
- Introduce **data cleaning tools** for handling missing or inconsistent data directly within the application.

Thank you for checking out **InsightMate**! 🙌 If you have any questions, feel free to open an issue or contribute to the project!

