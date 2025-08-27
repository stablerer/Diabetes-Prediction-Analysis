# Diabetes-Prediction-Analysis
A data analysis project for diabetes prediction using machine learning.
# Diabetes Prediction Analysis

## Project Overview
This project analyzes the Pima Indians Diabetes dataset to identify key clinical factors influencing diabetes onset using a Machine Learning approach. The goal is to demonstrate how data science can validate clinical knowledge and assist in early screening.

## Key Insights & Results
-   **Model Performance**: Achieved **76.62% accuracy** with a Random Forest classifier.
-   **Top Predictive Factors**: 
    1.  **Glucose** (0.25)
    2. **BMI**: (0.15)
    3.  **Age** (0.12)
    *The result is highly consistent with medical knowledge.*
-   **Population Health Insight**: 
    -   **38.7%** of the population in the dataset is diabetic.
    -   **36.3%** is pre-diabetic, highlighting a significant at-risk population.

## Technical Steps
1.  **Data Preprocessing**: Handled missing values (zeros in physiological measurements).
2.  **Exploratory Data Analysis (EDA)**: Visualized the distribution of key features.
3.  **Machine Learning**: Built and evaluated a Random Forest model.
4.  **Model Interpretation**: Analyzed feature importance.

## How to Run
1. **Clone** this repository.
2.  Install dependencies: `pip install -r requirements.txt` (See below for required packages)
3.  Open and run the Jupyter Notebook: `Diabetes_Prediction_Analysis.ipynb`

## Required Python Packages
-   pandas
-   numpy
-   scikit-learn
-   matplotlib

## File Structure
```
Diabetes-Prediction-Analysis/
├── Diabetes_Prediction_Analysis.ipynb  # Main analysis notebook
├── glucose_distribution.png            # Visualization output
├── feature_importance.png              # Visualization output
├── glucose_cdf.png                     # Cumulative distribution plot
├── glucose_pie_chart.png               # Population classification pie chart
└── README.md                           # Project description (this file)
```
