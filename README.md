# Clinical Trial Disease Classification Using NLP and Machine Learning

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify clinical trials into eight disease categories based on their brief descriptions.

The application allows users to enter a clinical trial summary and predict its corresponding disease category through an interactive Streamlit interface.

## Objectives

- Analyze clinical trial descriptions using NLP techniques.
- Preprocess and clean textual data.
- Build and compare machine learning classification models.
- Identify the most suitable model for disease classification.
- Develop an interactive web application for predictions and analysis.

## Dataset

- Total records: 60,337
- Number of features: 16
- Number of disease categories: 8

### Disease Categories

1. Anxiety
2. Breast Cancer
3. Chronic Obstructive Pulmonary Disease (COPD)
4. COVID-19
5. Glaucoma
6. Rheumatoid Arthritis
7. Sickle Cell Anemia
8. Type 2 Diabetes

**Input feature:** Brief Summary

**Target variable:** Source Condition Query

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- spaCy
- TF-IDF Vectorization
- Linear Support Vector Classification (Linear SVC)
- Streamlit

## Methodology

### 1. Data Preprocessing
- Lowercasing
- Text cleaning
- Tokenization
- Stopword removal
- Lemmatization

### 2. Exploratory Data Analysis
- Disease category distribution
- Text length distribution
- Most frequent words
- Disease-wise word frequency
- Missing value analysis
- Study type and trial status analysis

### 3. Feature Extraction

TF-IDF Vectorization was used to convert text into numerical features.

### 4. Model Building

The following models were evaluated:

- Logistic Regression
- Linear Support Vector Classification (Linear SVC)

Hyperparameter tuning was performed to improve model performance.

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 94.66% |
| Linear SVC | 94.68% |
| Improved Logistic Regression | 94.97% |
| Improved Linear SVC | 95.25% |
| Tuned Linear SVC | **95.38%** |

The tuned Linear SVC model achieved the best test accuracy of approximately 95.38%.

## Streamlit Application Features

The application contains the following sections:

1. **Home** – Project introduction and overview.
2. **Disease Prediction** – Predicts the disease category from a clinical trial summary.
3. **Dataset Overview** – Displays dataset information and category distribution.
4. **Exploratory Data Analysis** – Visualizes important patterns in the dataset.
5. **Model Performance** – Displays classification metrics and confusion matrix.
6. **Project Insights** – Presents key findings and conclusions.

## Project Structure

```text
Clinical-Trial-Disease-Classification/
│
├── Streamlit_clinical.py
├── clinical_trial_disease_classifier.pkl
├── processed_text.pkl
├── clinical_trials_raw_patient2trial_conditions.csv
├── requirements.txt
└── README.md
```

## How to Run the Application

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project folder

```bash
cd clinical-trial-disease-classification
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### Step 5: Run the Streamlit application

```bash
streamlit run Streamlit_clinical.py
```

## Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be applied to clinical trial text classification.

The tuned Linear SVC model achieved approximately 95.38% accuracy, demonstrating promising classification performance across eight disease categories.

The Streamlit application provides an interactive interface for exploring the dataset, understanding model performance and predicting disease categories from clinical trial summaries.
