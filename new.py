import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io  # Added to use StringIO for df.info()

# Cache the data loading function for performance
@st.cache_data
def load_data():
    # Update the path if your CSV file is located elsewhere.
    df = pd.read_csv('ai_job_market_insights.csv')
    return df

# Load the dataset
df = load_data()

# App Title and Description
st.title("AI-Powered Job Market Insights")
st.write("""
This web app provides interactive insights into a synthetic snapshot of the modern job market with a focus on artificial intelligence (AI) and automation.
The dataset consists of 500 unique job listings with various features such as job title, industry, company size, AI adoption level, automation risk, required skills, salary, remote work feasibility, and job growth projection.
""")

# Sidebar Menu for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Overview", "Exploratory Data Analysis", "Visualizations"])

# ----------------------
# Page 1: Overview
# ----------------------
if page == "Overview":
    st.header("Dataset Overview")
    
    st.subheader("Data Snapshot")
    st.dataframe(df.head())
    
    st.subheader("Dataset Shape")
    st.write("Rows, Columns:", df.shape)
    
    st.subheader("Missing and Duplicate Values")
    st.write("Missing Values per Column:")
    st.write(df.isnull().sum())
    st.write("Total Duplicate Rows:", df.duplicated().sum())
    
    st.subheader("Descriptive Statistics (Numerical)")
    st.write(df.describe())

# ----------------------
# Page 2: Exploratory Data Analysis (EDA)
# ----------------------
elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("DataFrame Info")
    # Use StringIO to capture the DataFrame info output
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.subheader("Unique Values in Categorical Columns")
    for col in df.columns:
        if df[col].dtype == 'object':
            st.markdown(f"{col}")
            st.write("Unique Values:", df[col].unique())
            st.write("Number of Unique Values:", df[col].nunique())
            st.write("---")

# ----------------------
# Page 3: Visualizations
# ----------------------
elif page == "Visualizations":
    st.header("Visualizations")
    
    # Create a selectbox to choose which visualization to display
    plot_option = st.selectbox("Select Plot", 
                               ["Company Size Distribution", 
                                "Job Title Distribution", 
                                "AI Adoption by Industry", 
                                "Salary Trends by Industry", 
                                "Remote Work Analysis", 
                                "Job Growth Projection Analysis"])
    
    if plot_option == "Company Size Distribution":
        st.subheader("Distribution of Company Size (Pie Chart)")
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = ['yellow', 'orange', '#99ff99']
        df["Company_Size"].value_counts().plot(
            kind='pie',
            startangle=20,
            autopct='%1.1f%%',
            explode=(0.08, 0, 0),
            shadow=True,
            colors=colors,
            ax=ax
        )
        ax.set_ylabel("")
        ax.set_title("Distribution of Company Size")
        st.pyplot(fig)
    
    elif plot_option == "Job Title Distribution":
        st.subheader("Distribution of Job Titles (Pie Chart)")
        value_counts = df['Job_Title'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
        ax.set_title("Distribution of Job Titles")
        st.pyplot(fig)
    
    elif plot_option == "AI Adoption by Industry":
        st.subheader("AI Adoption Level Across Different Industries")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x='AI_Adoption_Level', hue='Industry', ax=ax)
        ax.set_title("AI Adoption Level Across Different Industries")
        ax.set_xlabel("AI Adoption Level")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    elif plot_option == "Salary Trends by Industry":
        st.subheader("Salary Across Different Industries with AI Adoption Level")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df, x='Industry', y='Salary_USD', hue='AI_Adoption_Level', marker='o', ax=ax, ci=None)
        ax.set_title("Salary Trends by Industry")
        ax.set_xlabel("Industry")
        ax.set_ylabel("Salary (USD)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    elif plot_option == "Remote Work Analysis":
        st.subheader("Remote Work Analysis")
        st.write("*Remote Friendly Distribution (Pie Chart):*")
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        colors = plt.get_cmap('Pastel1_r').colors
        df["Remote_Friendly"].value_counts().plot(
            kind='pie',
            startangle=20,
            autopct='%1.1f%%',
            explode=(0.08, 0),
            shadow=True,
            colors=colors,
            ax=ax1
        )
        ax1.set_ylabel("")
        ax1.set_title("Remote Friendly Distribution")
        st.pyplot(fig1)
        
        st.write("*Remote Work Availability by Industry (Count Plot):*")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x='Industry', hue='Remote_Friendly', ax=ax2)
        ax2.set_title("Remote Work Availability by Industry")
        ax2.set_xlabel("Industry")
        ax2.set_ylabel("Count")
        plt.xticks(rotation=70)
        st.pyplot(fig2)
    
    elif plot_option == "Job Growth Projection Analysis":
        st.subheader("Job Growth Projection Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='Job_Growth_Projection', hue='Industry', ax=ax)
        ax.set_title("Job Growth Projection Across Different Industries")
        ax.set_xlabel("Job Growth Projection")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Optionally, display the raw dataset at the bottom of the app
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)
