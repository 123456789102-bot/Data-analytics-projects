import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load AI models
summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small"
)

sentiment_analyzer = SentimentIntensityAnalyzer()

st.title("AI-Powered Financial News Summarizer")

url = st.text_input("Paste a financial news article URL:")

if st.button("Analyze"):
    if url:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs])

        if len(article_text.strip()) == 0:
            st.error("Could not extract article text. Please try another link.")
        else:
            summary = summarizer(
                article_text[:1000],
                max_length=120,
                min_length=40,
                do_sample=False
            )[0]["summary_text"]

            sentiment_score = sentiment_analyzer.polarity_scores(summary)["compound"]

            if sentiment_score >= 0.05:
                sentiment = "📈 Bullish"
            elif sentiment_score <= -0.05:
                sentiment = "📉 Bearish"
            else:
                sentiment = "⚖️ Neutral"

            st.subheader("Summary")
            st.write(summary)

            st.subheader("Market Sentiment")
            st.write(sentiment)
    else:
        st.error("Please paste a valid URL.")

    
