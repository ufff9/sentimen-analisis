import streamlit as st
import pandas as pd
import joblib
import re
import string
import os
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
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

    st.markdown("""
    Dashboard analisis sentimen review film
    menggunakan metode:

    - TF-IDF
    - Support Vector Machine (SVM)
    - Logistic Regression
    - Naive Bayes
    """)

    positive_count = (
        df['sentiment'] == 'positive'
    ).sum()

    negative_count = (
        df['sentiment'] == 'negative'
    ).sum()

    total = len(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Reviews",
            f"{total:,}"
        )

    with col2:
        st.metric(
            "Positive Reviews",
            f"{positive_count:,}"
        )

    with col3:
        st.metric(
            "Negative Reviews",
            f"{negative_count:,}"
        )

    st.divider()

    st.subheader(
        "Sentiment Distribution"
    )

    sentiment_counts = (
        df['sentiment']
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(6,6)
    )

    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    st.divider()

    st.subheader(
        "Model Comparison"
    )

    if os.path.exists(
        "results/model_comparison.png"
    ):

        st.image(
            "results/model_comparison.png",
            use_container_width=True
        )

    else:

        st.warning(
            "model_comparison.png belum tersedia"
        )

# =====================================
# DATASET
# =====================================

elif menu == "Dataset":

    st.header(
        "📊 Dataset Overview"
    )

    st.write(
        f"Total Dataset : {len(df):,}"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Positive WordCloud"
        )

        if os.path.exists(
            "results/wordcloud_positive.png"
        ):

            st.image(
                "results/wordcloud_positive.png"
            )

        else:

            st.warning(
                "wordcloud_positive.png tidak ditemukan"
            )

    with col2:

        st.subheader(
            "Negative WordCloud"
        )

        if os.path.exists(
            "results/wordcloud_negative.png"
        ):

            st.image(
                "results/wordcloud_negative.png"
            )

        else:

            st.warning(
                "wordcloud_negative.png tidak ditemukan"
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

        if os.path.exists(
            "results/confusion_matrix.png"
        ):

            st.image(
                "results/confusion_matrix.png",
                use_container_width=True
            )

        else:

            st.error(
                "confusion_matrix.png tidak ditemukan"
            )

    with col2:

        st.subheader(
            "ROC Curve"
        )

        if os.path.exists(
            "results/roc_curve.png"
        ):

            st.image(
                "results/roc_curve.png",
                use_container_width=True
            )

        else:

            st.error(
                "roc_curve.png tidak ditemukan"
            )

    st.divider()

    st.subheader(
        "Model Accuracy Comparison"
    )

    if os.path.exists(
        "results/model_comparison.png"
    ):

        st.image(
            "results/model_comparison.png",
            use_container_width=True
        )

# =====================================
# PREDICTION
# =====================================

elif menu == "Prediction":

    st.header(
        "🎥 Sentiment Prediction"
    )

    review = st.text_area(
        "Enter Movie Review"
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

            score = abs(
                model.decision_function(
                    vector
                )[0]
            )

            confidence = min(
                score * 20,
                100
            )

            if prediction[0] == 1:

                st.success(
                    "😊 Positive Review"
                )

            else:

                st.error(
                    "😠 Negative Review"
                )

            st.subheader(
                "Confidence Score"
            )

            st.progress(
                int(confidence)
            )

            st.write(
                f"{confidence:.2f}%"
            )

# =====================================
# ABOUT
# =====================================

elif menu == "About":

    st.header(
        "ℹ️ About Project"
    )

    st.markdown("""
    ## IMDb Sentiment Analysis

    ### Dataset
    IMDb 50K Movie Reviews Dataset

    ### Feature Extraction
    TF-IDF

    ### Machine Learning Models
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Naive Bayes

    ### Best Model
    Support Vector Machine (SVM)

    ### Tools
    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - Streamlit
    - Matplotlib

    ### Author
    Machine Learning Project for Sentiment Analysis
    """)