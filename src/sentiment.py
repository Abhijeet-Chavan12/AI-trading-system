import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Handle optional Streamlit secrets
try:
    import streamlit as st
    NEWS_API_KEY = st.secrets["newsapi_key"]
except Exception:
    # Fallback API key
    NEWS_API_KEY = "e6eb0e24c63a4a799dacd463d7935937"

analyzer = SentimentIntensityAnalyzer()

def fetch_company_news_sentiment(company_name, max_articles=10):
    """
    Fetch news from NewsAPI and calculate average sentiment.
    Returns: sentiment_label, sentiment_score, list of article dicts with sentiment.
    """
    url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])[:max_articles]

        scored_articles = []
        total_score = 0
        for article in articles:
            text = article.get("title", "") + ". " + article.get("description", "")
            score = analyzer.polarity_scores(text)["compound"]
            total_score += score
            scored_articles.append({
                "title": article["title"],
                "url": article["url"],
                "score": score
            })

        avg_score = total_score / len(scored_articles) if scored_articles else 0

        if avg_score > 0.05:
            sentiment = "Positive"
        elif avg_score < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return sentiment, avg_score, scored_articles

    except Exception as e:
        print(f"❌ Sentiment fetch error: {e}")
        return "Neutral", 0.0, []
