import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to database
conn = sqlite3.connect('db/tweets.db')

# Query 1: Total tweets per day
query_daily = """
SELECT DATE(date) AS tweet_date, COUNT(*) AS tweet_count
FROM tweets
GROUP BY tweet_date
ORDER BY tweet_date;
"""
df_daily = pd.read_sql(query_daily, conn)
print("Tweets per day:")
print(df_daily.head())

# Query 2: Top 10 user locations
query_locations = """
SELECT user_location, COUNT(*) AS tweet_count
FROM tweets
WHERE user_location != 'Unknown'
GROUP BY user_location
ORDER BY tweet_count DESC
LIMIT 10;
"""
df_locations = pd.read_sql(query_locations, conn)
print("\n Top 10 user locations:")
print(df_locations)

# Line chart: Tweet count over time
plt.figure(figsize=(10, 5))
plt.plot(df_daily['tweet_date'], df_daily['tweet_count'], marker='o')
plt.title('Squid Game Tweet Activity Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
conn.close()
