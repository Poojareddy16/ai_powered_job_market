# 💼 AI-Powered Job Market Insights Dashboard

A Streamlit-based interactive web application to explore and visualize trends in the modern AI-driven job market. The app analyzes a dataset of synthetic job listings to uncover insights into automation risk, AI adoption, salaries, remote work, and more.

## 📌 Overview

This project enables users to:

- Analyze job trends and AI adoption levels across industries.
- Visualize salary distributions and job growth projections.
- Explore company size, job titles, and remote work opportunities.
- Perform basic EDA and inspect the dataset interactively.

## 🛠 Features

### 🧭 Navigation Menu

The sidebar allows users to switch between:
- **Overview**: View a snapshot of the dataset, its structure, and basic stats.
- **Exploratory Data Analysis**: Dive into data types, categorical values, and distributions.
- **Visualizations**: Generate various interactive plots to uncover deeper insights.

### 📊 Visualization Options

- **Company Size Distribution** (Pie Chart)
- **Job Title Distribution** (Pie Chart)
- **AI Adoption by Industry** (Count Plot)
- **Salary Trends by Industry & AI Level** (Line Plot)
- **Remote Work Analysis** (Pie + Count Plot)
- **Job Growth Projections** (Grouped Bar Plot)

## 📂 Dataset Information

- **File**: `ai_job_market_insights.csv`
- **Records**: 500 job listings (synthetic data)
- **Attributes**:
  - `Job_Title`
  - `Industry`
  - `Company_Size`
  - `AI_Adoption_Level`
  - `Automation_Risk`
  - `Required_Skills`
  - `Salary_USD`
  - `Remote_Friendly`
  - `Job_Growth_Projection`

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Seaborn, Matplotlib, Plotly Express
- **Others**: `io.StringIO` for rendering `df.info()`

---

## 🚀 How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-job-market-insights.git
cd ai-job-market-insights
```

### 2. Install Dependencies
Make sure you have Python 3.7+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Add Dataset
Place the `ai_job_market_insights.csv` file in the project root directory or update the path in `load_data()` if needed.

### 4. Launch the App
```bash
streamlit run new.py
```
