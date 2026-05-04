import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if pd.isna(text) or text == '':
        return 0.0
    return analyzer.polarity_scores(str(text))['compound']

def count_hashtags(text):
    if pd.isna(text):
        return 0
    return str(text).count('#')

def compute_virality_label(df, likes_col, comments_col, followers_col, percentile=0.80):
    df = df.copy()
    df['engagement_rate'] = (df[likes_col] + df[comments_col]) / df[followers_col]
    threshold = df['engagement_rate'].quantile(percentile)
    df['viral'] = (df['engagement_rate'] > threshold).astype(int)
    print(f"Threshold (p{int(percentile*100)}): {threshold:.4f}")
    print(df['viral'].value_counts())
    return df
