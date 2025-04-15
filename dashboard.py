import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, pearsonr, chi2_contingency
import plotly.express as px


# Connect to DB and load data
conn = sqlite3.connect('db/tweets.db')
df = pd.read_sql("SELECT * FROM tweets", conn)
conn.close()

#  Tweet activity per day
df_daily = df.groupby(df['date'].str[:10]).size().reset_index(name='tweet_count')
df_daily = df_daily.rename(columns={'date': 'tweet_date'})

#  Top 10 locations
df_locations = df[df['user_location'] != 'unknown'] \
    .groupby('user_location') \
    .size() \
    .reset_index(name='tweet_count') \
    .sort_values(by='tweet_count', ascending=False) \
    .head(10)

#  Top 10 hashtags
all_hashtags = []
for tags in df['hashtags']:
    if pd.notna(tags):
        for tag in tags.split(','):
            clean_tag = tag.strip()
            if clean_tag:
                all_hashtags.append(clean_tag)

top_tags = Counter(all_hashtags).most_common(10)
tag_df = pd.DataFrame(top_tags, columns=['Hashtag', 'Count'])

#  Average sentiment over time
df['date_short'] = df['date'].str[:10]
df_sentiment = df.groupby('date_short')['sentiment_score'].mean().reset_index()
df_sentiment = df_sentiment.rename(columns={'date_short': 'tweet_date'})

#  Streamlit layout
st.title(" Squid Game Tweet Insights Dashboard")

#input image traffic light game
image = Image.open("images/squidgame_trafficlights.jpg")
st.image(image, caption="무궁화 꽃이 피었습니다", use_container_width=True)



# Tweets per day
st.subheader("Tweet Activity Over Time")
st.line_chart(df_daily.set_index('tweet_date'))

# Locations - Horizontal bar
st.subheader("Top 10 Tweet Locations (Horizontal Bar)")
fig1, ax1 = plt.subplots()
ax1.barh(df_locations['user_location'], df_locations['tweet_count'], color='pink')
ax1.set_xlabel("Tweet Count")
ax1.set_ylabel("Location")
ax1.set_title("Top 10 Tweet Locations")
ax1.invert_yaxis()
st.pyplot(fig1)

# Hashtags - Horizontal bar
st.subheader(" Top 10 Hashtags (Horizontal Bar)")
fig2, ax2 = plt.subplots()
ax2.barh(tag_df['Hashtag'], tag_df['Count'], color='lightgreen')
ax2.set_xlabel("Count")
ax2.set_ylabel("Hashtag")
ax2.set_title("Top 10 Hashtags")
ax2.invert_yaxis()
st.pyplot(fig2)

# Sentiment over time
st.subheader(" Average Tweet Sentiment Over Time")
fig3, ax3 = plt.subplots()
ax3.plot(df_sentiment['tweet_date'], df_sentiment['sentiment_score'], marker='o', color='purple')
ax3.set_ylabel("Average Sentiment Score")
ax3.set_xlabel("Date")
ax3.set_title("Tweet Sentiment Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)


#Start from here is K-Means Clstering,group users by behavior

st.subheader(" K-Means: Clustering Users by Behavior")

# Choose features
df_k = df[['user_followers', 'user_friends', 'sentiment_score']].dropna()

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_k)

# Start here is K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df_k['cluster'] = kmeans.fit_predict(X_scaled)

fig4, ax4 = plt.subplots()
scatter = ax4.scatter(df_k['user_followers'], df_k['sentiment_score'], c=df_k['cluster'], cmap='viridis')
ax4.set_xlabel('Followers')
ax4.set_ylabel('Sentiment Score')
ax4.set_title('User Clusters: Followers vs Sentiment')
st.pyplot(fig4)

#Start from here is T-test
st.subheader(" T-Test: Sentiment Between Verified and Non-Verified Users")

verified = df[df['user_verified'] == True]['sentiment_score']
unverified = df[df['user_verified'] == False]['sentiment_score']

t_stat, p_val = ttest_ind(verified, unverified, equal_var=False)

st.write(f"T-statistic: {t_stat:.3f}")
st.write(f"P-value: {p_val:.3f}")

if p_val < 0.05:
    st.success("Significant difference in sentiment between groups")
else:
    st.info("No significant difference in sentiment")


#Start from here is Pearson Correlation
st.subheader("Pearson Correlation Matrix")

# Calculate correlations
corr_cols = ['user_followers', 'user_friends', 'sentiment_score']
corr_matrix = df[corr_cols].corr()

fig5, ax5 = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax5)
ax5.set_title("Pearson Correlation Matrix")
st.pyplot(fig5)

# Example: followers vs sentiment
r, p = pearsonr(df['user_followers'], df['sentiment_score'])
st.write(f"Pearson r (followers vs sentiment): {r:.3f}, p = {p:.3f}")


#Monte Carlo Simulation
st.subheader("Monte Carlo Simulation: Predicting Avg Sentiment")

# Simulate avg sentiment from 10,000 random tweet samples
simulations = []
for _ in range(10000):
    sample = df['sentiment_score'].dropna().sample(100, replace=True)
    simulations.append(sample.mean())

# Plot simulation results
fig6, ax6 = plt.subplots()
ax6.hist(simulations, bins=50, color='orange', alpha=0.7)
ax6.set_title("Monte Carlo Simulation: Average Sentiment (n=100)")
ax6.set_xlabel("Average Sentiment Score")
st.pyplot(fig6)

st.write(f"Estimated Mean Sentiment: {np.mean(simulations):.3f}")
st.write(f"95% CI: {np.percentile(simulations, 2.5):.3f} to {np.percentile(simulations, 97.5):.3f}")

#Heat Map

st.subheader("Tweet Distribution by Country")

# Clean + prep country-level tweet counts
country_counts = df['country'].value_counts().reset_index()
country_counts.columns = ['country', 'tweet_count']


# Plot map
fig_map = px.choropleth(
    country_counts,
    locations='country',
    locationmode='country names',
    color='tweet_count',
    range_color=(0, 1000),
    color_continuous_scale='reds',
    title='Tweet Volume by Country'
)
st.plotly_chart(fig_map)

#Chi-Square Test
st.subheader("Chi-Square Test: Sentiment vs Verified Status")

# Create contingency table
contingency = pd.crosstab(df['user_verified'], df['sentiment_label'])

# Run chi-square
chi2, p, dof, expected = chi2_contingency(contingency)

st.write("Contingency Table:")
st.dataframe(contingency)
st.write(f"Chi-Square: {chi2:.4f}")
st.write(f"P-value: {p:.4f}")

if p < 0.05:
    st.success("Sentiment distribution differs significantly between verified and non-verified users")
else:
    st.info("No significant difference in sentiment distribution")

