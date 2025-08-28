# import packages
import os
os.environ["USE_TF"] = "0"   # disable TensorFlow usage
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Load Hugging Face sentiment pipeline (runs locally, downloads model first time)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_sentiment_model()

# Function to classify sentiment locally
@st.cache_data
def get_sentiment(text):
    if not text or (isinstance(text, float) and pd.isna(text)):
        return "Neutral"
    try:
        result = sentiment_pipeline(text[:512])[0]  # truncate long text
        label = result["label"].upper()
        if "POSITIVE" in label:
            return "Positive"
        elif "NEGATIVE" in label:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"Model error: {e}")
        return "Neutral"

# Streamlit UI
st.title("🔍 Local Hugging Face Sentiment Analysis Dashboard")
st.write("Upload a dataset and analyze sentiments locally (no API quota needed).")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]  # clean column names
    st.session_state["df"] = df.copy()

    st.success("✅ Dataset uploaded.")
    st.write("Preview:")
    st.dataframe(df.head(50))

    # Pick column to analyze
    text_col = st.selectbox("📝 Choose the text column to analyze", df.columns)

    if st.button("🔍 Analyze Sentiment"):
        try:
            with st.spinner("Analyzing sentiment locally..."):
                st.session_state["df"]["Sentiment"] = st.session_state["df"][text_col].apply(get_sentiment)
            st.success("🎉 Sentiment analysis completed!")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    # Show analyzed data
    if "Sentiment" in st.session_state["df"].columns:
        st.subheader("📊 Sentiment Breakdown")

        # Optional product filter if PRODUCT column exists
        if "PRODUCT" in st.session_state["df"].columns:
            product_options = ["All Products"] + sorted(st.session_state["df"]["PRODUCT"].dropna().unique().tolist())
            product = st.selectbox("Filter by PRODUCT (optional)", product_options)
            if product != "All Products":
                filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
            else:
                filtered_df = st.session_state["df"]
        else:
            filtered_df = st.session_state["df"]

        st.dataframe(filtered_df.head(100))

        # Bar chart
        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        sentiment_order = ["Negative", "Neutral", "Positive"]
        sentiment_counts["Sentiment"] = pd.Categorical(
            sentiment_counts["Sentiment"], categories=sentiment_order, ordered=True
        )
        sentiment_counts = sentiment_counts.sort_values("Sentiment")

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Distribution of Sentiment Classifications",
            labels={"Sentiment": "Sentiment Category", "Count": "Number of Reviews"},
            color="Sentiment",
            color_discrete_map={"Negative": "red", "Neutral": "lightgray", "Positive": "green"},
        )
        fig.update_layout(xaxis_title="Sentiment Category", yaxis_title="Number of Reviews", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Upload a CSV to begin.")
