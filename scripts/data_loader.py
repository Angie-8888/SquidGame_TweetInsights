import pandas as pd
import sqlite3
import os
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Set paths
csv_path = os.path.join('data', 'tweets_v8.csv')
db_path = os.path.join('db', 'tweets.db')

# Load CSV
print("Loading CSV...")
df = pd.read_csv(csv_path)

# Clean missing data
df['user_location'] = df['user_location'].fillna('Unknown')
df['date'] = pd.to_datetime(df['date'])

# Clean and standardize locations
df['user_location'] = df['user_location'].str.lower().str.strip()
location_map = {
    'nyc': 'new york',
    'n.y.c.': 'new york',
    'new york, ny': 'new york',
    'new york city': 'new york',

    'la': 'los angeles',
    'l.a.': 'los angeles',
    'los angeles, ca': 'los angeles',

    'london, england': 'london',
    'england, united kingdom': 'united kingdom',
    'uk': 'united kingdom',
    'u.k.': 'united kingdom',

    'usa': 'united states',
    'us': 'united states',
    'u.s.': 'united states',
}
# Apply partial matching to clean user_location
def map_location(loc):
    for key, value in location_map.items():
        if key in loc:
            return value
    return loc

df['user_location'] = df['user_location'].apply(map_location)

# Basic keyword-to-country mapping 
country_keywords = {
    'united states': 'United States',
    'usa': 'United States',
    'canada': 'Canada',
    'uk': 'United Kingdom',
    'united kingdom': 'United Kingdom',
    'india': 'India',
    'australia': 'Australia',
    'france': 'France',
    'germany': 'Germany',
    'brazil': 'Brazil',
    'philippines': 'Philippines',
    'nigeria': 'Nigeria',
    'japan': 'Japan',
    'malaysia': 'Malaysia',
    'mexico': 'Mexico'
}

# Map country names based on a substring match
def extract_country(location):
    for keyword, country in country_keywords.items():
        if keyword in location:
            return country
    return 'Other'

df['country'] = df['user_location'].apply(extract_country)

# Hashtag extraction
def extract_hashtags(text):
    return re.findall(r"#\w+", str(text).lower())

df['hashtags'] = df['text'].apply(lambda x: ', '.join(extract_hashtags(x)))

# Sentiment analysis using VADER
sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

#Chi-Square Test
def label_sentiment(score):
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['sentiment_score'].apply(label_sentiment)

# Save to SQLite
print("Saving to SQLite database...")
conn = sqlite3.connect(db_path)
df.to_sql('tweets', conn, if_exists='replace', index=False)

print("Data successfully saved to tweets.db!")
conn.close()
