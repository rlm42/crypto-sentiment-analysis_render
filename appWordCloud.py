import tweepy
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# enter your API credentials
access_key = "7j9aAfN6PXNJCJ4wm7We2co3R"
access_secret = "E0MbVMJ1fGIJTLmBPZ02iMQRRpTjS6djYStFlDL6W411fGQgAW"
consumer_key = "1438071016169742337-Zd0vl4jYArjx0xD6rIOwyEG2JgfKvn"
consumer_secret = "pBzsBi3PmTDlfiSyaztF7RLod60o7cyTo95W1glSLb0Sq"

# Twitter authentication
auth = tweepy.OAuthHandler(access_key, access_secret)   
auth.set_access_token(consumer_key, consumer_secret) 
api = tweepy.API(auth)

# set the search query
query = "#crypto"

# set the number of tweets to scrape
num_tweets = 300

# scrape the tweets
tweets = tweepy.Cursor(api.search_tweets,
                       q=query,
                       lang="en",
                       tweet_mode="extended").items(num_tweets)

# store the tweets in a pandas DataFrame
data = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=["text"])

# define a function to clean the text data
def clean_text(text):
    # remove URLs
    text = re.sub(r"http\S+", "", text)
    # remove special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # convert to lowercase
    text = text.lower()
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# apply the clean_text function to the text column
data["text_clean"] = data["text"].apply(clean_text)

# extract symbols starting with $ or #
data["symbols"] = data["text"].apply(lambda x: [symbol for symbol in re.findall(r"[$#][a-zA-Z]+", x) if symbol.lower() not in ["#crypto", "#giveaway"]])

# count the occurrence of each symbol
symbol_counts = Counter([symbol for symbols in data["symbols"] for symbol in symbols])

# smallest trending largest first   COMMENT THIS OUT IF WANT ORIGINAL IMAGE
symbol_counts_inverted = {symbol: 1/count for symbol, count in symbol_counts.items()}

# create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color="white", margin=0)

# generate the word cloud from the symbol counts (with smallest first)  REMOVE _inverted IF WANT ORIGINAL IMAGE
wordcloud.generate_from_frequencies(symbol_counts_inverted)

# save the word cloud as an SVG file
wordcloud.to_file("static/crypto_wordcloud.png")

# get the current directory of the Python script
dir_path = os.path.dirname(os.path.realpath(__file__))

# create the path to the static folder
static_path = os.path.join(dir_path, 'static')

# set the file path to save the word cloud image
file_path = os.path.join(static_path, 'word_cloud.svg')

# plot the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# save the word cloud image
plt.savefig(file_path, dpi=700, format="svg")
plt.show()

