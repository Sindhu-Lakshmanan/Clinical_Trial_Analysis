import streamlit as st
import pandas as pd
import joblib
import re
import spacy
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split
# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Clinical Trial Disease Classifier",
    page_icon="🧬",
    layout="wide"
)


# --------------------------------
# 2. Load Model
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("clinical_trial_disease_classifier.pkl")


model = load_model()


# --------------------------------
# 3. Load spaCy
# --------------------------------

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")


nlp = load_spacy()


# --------------------------------
# 4. Load Dataset
# --------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(
        r"C:\Users\sindh\OneDrive\Documents\Data science GUVI\project 5\clinical_trials_raw_patient2trial_conditions.csv"
    )


clinic_df = load_dataset()


# --------------------------------
# 5. Text Preprocessing
# --------------------------------


stop_words = set(ENGLISH_STOP_WORDS)

negation_words = {
    'no', 'not', 'never', 'neither', 'nor', 'without'
}

stop_words = stop_words - negation_words
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words and word != '-'
    ]

    doc = nlp(" ".join(tokens))

    lemmatized_tokens = [
        token.lemma_
        for token in doc
        if token.text != '-'
    ]

    return " ".join(lemmatized_tokens)
# Load existing processed text from Jupyter Notebook

@st.cache_data
def load_processed_text():
    return pd.read_pickle("processed_text.pkl")

processed_text = load_processed_text()


# --------------------------------
# 6. Sidebar Navigation
# --------------------------------

st.sidebar.title("🧬 Clinical Trial Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Disease Prediction",
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Model Performance",
        "Project Insights"
    ]
)
# --------------------------------
# 7. Home Page
# --------------------------------

if page == "Home":

    st.title("🧬 Clinical Trial Disease Classification")

    st.write(
        "An NLP and Machine Learning application "
        "to classify clinical trials into disease categories."
    )

    st.markdown("---")

    st.subheader("Project Objective")

    st.write(
        "The objective of this project is to analyze "
        "clinical trial brief summaries and predict "
        "their corresponding disease categories."
    )

    st.subheader("Machine Learning Model")

    st.write("TF-IDF Vectorization + Tuned Linear SVC")

    st.subheader("Disease Categories")

    st.write("The model classifies clinical trials into 8 disease categories.")


# --------------------------------
# 8. Disease Prediction Page
# --------------------------------

elif page == "Disease Prediction":

    st.title("🔍 Disease Prediction")

    st.write(
        "Enter a clinical trial Brief Summary "
        "to predict its disease category."
    )

    user_input = st.text_area(
        "Clinical Trial Brief Summary",
        height=200,
        placeholder="Enter the clinical trial summary here..."
    )

    if st.button("Predict Disease"):

        if user_input.strip() == "":

            st.warning("Please enter a clinical trial summary.")

        else:

            with st.spinner("Analyzing clinical trial summary..."):

                processed_input = preprocess_text(user_input)

                prediction = model.predict([processed_input])

            st.success("Prediction Completed!")

            st.subheader("Predicted Disease Category")

            st.markdown(f"## 🏥 {prediction[0].title()}")


# --------------------------------
# 9. Dataset Overview Page
# --------------------------------

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Clinical Trials",
            f"{len(clinic_df):,}"
        )

    with col2:
        st.metric(
            "Total Features",
            clinic_df.shape[1]
        )

    with col3:
        st.metric(
            "Disease Categories",
            clinic_df['source_condition_query'].nunique()
        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        clinic_df.head(10),
        use_container_width=True
    )

    st.subheader("Disease Category Distribution")

    disease_counts = clinic_df[
        'source_condition_query'
    ].value_counts()

    st.bar_chart(disease_counts)

    st.subheader("Disease Category Counts")

    st.dataframe(
        disease_counts.rename_axis(
            "Disease Category"
        ).reset_index(
            name="Number of Clinical Trials"
        ),
        use_container_width=True
    )
# --------------------------------
# 10. Exploratory Data Analysis
# --------------------------------

elif page == "Exploratory Data Analysis":

    st.title("📈 Exploratory Data Analysis")

    # 1. Disease Category Distribution
    st.subheader("1. Disease Category Distribution")

    disease_counts = clinic_df['source_condition_query'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 5))
    disease_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel("Disease Category")
    ax.set_ylabel("Number of Clinical Trials")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    # 2. Class Imbalance
    st.subheader("2. Disease Category Proportion")

    fig, ax = plt.subplots(figsize=(8, 8))
    disease_counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close(fig)

    # 3. Brief Summary Word Count Distribution
    st.subheader("3. Brief Summary Word Count Distribution")

    text_lengths = clinic_df['brief_summary'].astype(str).apply(
        lambda text: len(text.split())
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(text_lengths, bins=50)
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)

    # 4. Text Length by Disease
    st.subheader("4. Text Length by Disease Category")

    length_df = pd.DataFrame({
        'Disease': clinic_df['source_condition_query'],
        'Text Length': text_lengths
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    length_df.boxplot(
        column='Text Length',
        by='Disease',
        ax=ax,
        rot=45
    )

    ax.set_title("Text Length Distribution by Disease")
    ax.set_xlabel("Disease Category")
    ax.set_ylabel("Number of Words")
    fig.suptitle("")
    st.pyplot(fig)
    plt.close(fig)

    # 5. Average Text Length
    st.subheader("5. Average Text Length by Disease")

    avg_length = length_df.groupby('Disease')['Text Length'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    avg_length.sort_values().plot(kind='bar', ax=ax)
    ax.set_ylabel("Average Number of Words")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    # 6. Most Frequent Words Overall

    st.subheader("6. Most Frequent Words in Clinical Trial Summaries")

    all_words = []

    for text in processed_text:
        all_words.extend(text.split())

    word_counts = Counter(all_words)

    top_words = pd.Series(
        dict(word_counts.most_common(20))
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    top_words.sort_values().plot(kind='barh', ax=ax)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Words")

    st.pyplot(fig)
    plt.close(fig)

    # 7. Disease-wise Top 10 Words

    st.subheader("7. Disease-wise Top 10 Words")

    for disease in clinic_df['source_condition_query'].unique():

        disease_texts = processed_text[
            clinic_df['source_condition_query'] == disease
        ]

        words = []

        for text in disease_texts:
            words.extend(text.split())

        disease_word_counts = Counter(words)

        top_disease_words = pd.Series(
            dict(disease_word_counts.most_common(10))
        )

        st.write(f"**{disease.title()}**")

        fig, ax = plt.subplots(figsize=(10, 4))

        top_disease_words.sort_values().plot(
            kind='barh',
            ax=ax
        )

        ax.set_xlabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)
        # 8. Missing Value Analysis
        st.subheader("8. Missing Value Analysis")

        missing_values = clinic_df.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values()

        if len(missing_values) > 0:

            fig, ax = plt.subplots(figsize=(12, 5))
            missing_values.plot(kind='barh', ax=ax)
            ax.set_xlabel("Number of Missing Values")
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.success("No missing values found.")

        # 9. Study Type Distribution
        st.subheader("9. Clinical Trial Study Type Distribution")

        if 'study_type' in clinic_df.columns:

            study_counts = clinic_df['study_type'].value_counts()

            st.bar_chart(study_counts)

        # 10. Overall Status Distribution
        st.subheader("10. Clinical Trial Status Distribution")

        if 'overall_status' in clinic_df.columns:

            status_counts = clinic_df['overall_status'].value_counts()

            st.bar_chart(status_counts)

    # 11. Clinical Trial Phase Distribution
    st.subheader("11. Clinical Trial Phase Distribution")

    if 'phase' in clinic_df.columns:

        phase_counts = clinic_df['phase'].value_counts()

        st.bar_chart(phase_counts)
# --------------------------------
# Model Performance
# --------------------------------

elif page == "Model Performance":

    st.title("🎯 Model Performance")

    @st.cache_data
    def evaluate_model():

        X = processed_text
        y = clinic_df['source_condition_query']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=model.classes_
        )

        return accuracy, report, cm, y_test, y_pred

    accuracy, report, cm, y_test, y_pred = evaluate_model()

    # Accuracy
    st.subheader("1. Overall Accuracy")

    st.metric(
        "Tuned Linear SVC Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    # Classification Report
    st.subheader("2. Classification Report")

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # Confusion Matrix
    st.subheader("3. Confusion Matrix")

    fig, ax = plt.subplots(figsize=(12, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        values_format="d"
    )

    st.pyplot(fig)
    plt.close(fig)

    # Correct and Incorrect Predictions
    st.subheader("4. Prediction Summary")

    correct = (y_test == y_pred).sum()
    incorrect = (y_test != y_pred).sum()

    col1, col2 = st.columns(2)

    col1.metric("Correct Predictions", f"{correct:,}")
    col2.metric("Incorrect Predictions", f"{incorrect:,}")
# --------------------------------
# Project Insights & Conclusion
# --------------------------------

elif page == "Project Insights":

    st.title("📌 Project Insights & Conclusion")

    st.subheader("1. Dataset Insights")

    st.markdown("""
    - The dataset contains 60,337 clinical trials.
    - The dataset covers 8 disease categories.
    - Breast cancer has the highest number of records.
    - Sickle cell anemia has the lowest number of records.
    - The dataset has an imbalanced class distribution.
    """)

    st.markdown("---")

    st.subheader("2. NLP Insights")

    st.markdown("""
    - Text preprocessing included lowercasing, cleaning,
      tokenization, stopword removal and lemmatization.
    - TF-IDF was used to convert text into numerical features.
    - Unigrams and bigrams were considered during feature extraction.
    """)

    st.markdown("---")

    st.subheader("3. Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Linear SVC",
            "Improved Logistic Regression",
            "Improved Linear SVC",
            "Tuned Linear SVC"
        ],
        "Accuracy (%)": [
            94.66,
            94.68,
            94.97,
            95.25,
            95.38
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("---")

    st.subheader("4. Key Findings")

    st.markdown("""
    - Tuned Linear SVC achieved the best test accuracy of 95.38%.
    - Five-fold cross-validation achieved an average accuracy of 94.90%.
    - Sickle cell anemia had comparatively lower recall than
      the other disease categories.
    - The model demonstrated strong classification performance
      across all eight disease categories.
    """)

    st.markdown("---")

    st.subheader("5. Conclusion")

    st.write("""
    This project demonstrates the application of Natural Language
    Processing and Machine Learning for clinical trial disease
    classification.

    Using TF-IDF feature extraction and a tuned Linear SVC model,
    the system achieved 95.38% accuracy on the held-out test dataset.

    The developed Streamlit application provides an interactive
    interface for predicting disease categories from clinical
    trial brief summaries.
    """)