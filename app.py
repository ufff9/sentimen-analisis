import streamlit as st
import pandas as pd
import joblib
import re
import string
import os

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/svm_model.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_model()

# =====================================
# LOAD DATASET
# =====================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/IMDB Dataset.csv"
    )

df = load_data()

# =====================================
# PREPROCESSING
# =====================================

def preprocessing(text):

    text = text.lower()

    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    text = re.sub(
        r'https?://\S+',
        '',
        text
    )

    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    return text

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🎬 Navigation")

menu = st.sidebar.radio(
    "Choose Menu",
    [
        "Home",
        "Dataset",
        "Evaluation",
        "Prediction",
        "About"
    ]
)

# =====================================
# HOME
# =====================================

if menu == "Home":

    st.title(
        "🎬 IMDb Sentiment Analysis Dashboard"
    )

    st.markdown(
        """
        Dashboard untuk analisis sentimen review film IMDb
        menggunakan:

        - TF-IDF
        - Support Vector Machine (SVM)
        - Logistic Regression
        - Naive Bayes
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Reviews",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Positive Reviews",
            f"{(df['sentiment']=='positive').sum():,}"
        )

    with col3:
        st.metric(
            "Negative Reviews",
            f"{(df['sentiment']=='negative').sum():,}"
        )

    st.image(
        "results/model_comparison.png",
        caption="Model Comparison"
    )

# =====================================
# DATASET
# =====================================

elif menu == "Dataset":

    st.header("📊 Dataset Overview")

    st.write(
        f"Jumlah Data: {len(df):,}"
    )

    st.dataframe(
        df.head(20)
    )

    st.subheader(
        "WordCloud Positive"
    )

    st.image(
        "results/wordcloud_positive.png"
    )

    st.subheader(
        "WordCloud Negative"
    )

    st.image(
        "results/wordcloud_negative.png"
    )

# =====================================
# EVALUATION
# =====================================

elif menu == "Evaluation":

    st.header(
        "📈 Model Evaluation"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Confusion Matrix"
        )

        st.image(
            "results/confusion_matrix.png"
        )

    with col2:

        st.subheader(
            "ROC Curve"
        )

        st.image(
            "results/roc_curve.png"
        )

    st.subheader(
        "Accuracy Comparison"
    )

    st.image(
        "results/model_comparison.png"
    )

# =====================================
# PREDICTION
# =====================================

elif menu == "Prediction":

    st.header(
        "🎥 Predict Review Sentiment"
    )

    review = st.text_area(
        "Input Movie Review"
    )

    if st.button(
        "Analyze Sentiment"
    ):

        if review.strip() == "":

            st.warning(
                "Masukkan review terlebih dahulu."
            )

        else:

            clean_text = preprocessing(
                review
            )

            vector = vectorizer.transform(
                [clean_text]
            )

            prediction = model.predict(
                vector
            )

            if prediction[0] == 1:

                st.success(
                    "😊 Positive Review"
                )

            else:

                st.error(
                    "😠 Negative Review"
                )

# =====================================
# ABOUT
# =====================================

elif menu == "About":

    st.header(
        "ℹ️ About Project"
    )

    st.markdown(
        """
        ### Dataset

        IMDb Movie Reviews Dataset

        ### Models

        - Logistic Regression
        - Support Vector Machine
        - Naive Bayes

        ### Best Model

        Support Vector Machine (SVM)

        ### Tools

        - Python
        - Scikit-Learn
        - Pandas
        - Streamlit
        """
    )