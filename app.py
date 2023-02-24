from flask import Flask, request, render_template
import tweepy
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
plt.matplotlib.use('Agg')  # Use Agg backend to avoid starting a Matplotlib GUI outside the main thread
import base64
from io import BytesIO
from langdetect import detect
import requests
from bs4 import BeautifulSoup
import torch.cuda
torch.cuda.set_device(0)  # Set the default device to be the first GPU

import re
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import os


app = Flask(__name__)

# Twitter API authentication
auth = tweepy.OAuthHandler("7j9aAfN6PXNJCJ4wm7We2co3R", "E0MbVMJ1fGIJTLmBPZ02iMQRRpTjS6djYStFlDL6W411fGQgAW")
auth.set_access_token("1438071016169742337-Zd0vl4jYArjx0xD6rIOwyEG2JgfKvn", "pBzsBi3PmTDlfiSyaztF7RLod60o7cyTo95W1glSLb0Sq")
api = tweepy.API(auth)


# THE FOLLOWING IS TO UPDATE THE WORD CLOUD

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

# smallest trending largest first   DON"T USE IF WANT ORIGINAL IMAGE
symbol_counts_inverted = {symbol: 1/count for symbol, count in symbol_counts.items()}

# create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color="white", margin=0)
 
# generate the word cloud from the symbol counts (with smallest first)  REMOVE _inverted IF WANT ORIGINAL IMAGE
wordcloud.generate_from_frequencies(symbol_counts)

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

# END OF UPDATING THE WORD CLOUD


# Sentiment analysis model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

config = AutoConfig.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        if t.startswith("@") and len(t) > 1:
            t = "@user"
        elif t.startswith("http"):
            t = "http"
        new_text.append(t)
    return " ".join(new_text)

def is_english(text):
    try:
        language = detect(text)
        if language == "en":
            return True
        else:
            return False
    except:
        return False

def analyze_sentiment(df):
    for i, row in df.iterrows():
        text = row["text"]
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors="pt").to(device)  # move the input tensor to device
        output = model(**encoded_input)
        scores = output.logits[0].detach().cpu().numpy()  # move the output tensor to CPU and convert to numpy
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        label = config.id2label[ranking[0]]
        score = scores[ranking[0]]
        df.at[i, "sentiment_label"] = label
        df.at[i, "sentiment_score"] = score
    return df

def analyze_sentiment_news(df_news):
    for i, row in df_news.iterrows():
        text = row["description"]
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors="pt").to(device)  # move the input tensor to device
        output = model(**encoded_input)
        scores = output.logits[0].detach().cpu().numpy()  # move the output tensor to CPU and convert to numpy
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        label = config.id2label[ranking[0]]
        score = scores[ranking[0]]
        df_news.at[i, "sentiment_label"] = label
        df_news.at[i, "sentiment_score"] = score
    return df_news





@app.route("/tweets", methods=["GET", "POST"])
def tweets():
    if request.method == "POST":
        ticker = request.form["ticker"]
        tweets = tweepy.Cursor(api.search_tweets, q=ticker, tweet_mode='extended').items(100)

        tweet_list = []
        # Create a set to keep track of the tweet content
        tweet_set = set()

        for tweet in tweets:
            text = tweet._json["full_text"]
                # Check if tweet is a retweet
            if hasattr(tweet, "retweeted_status"):
                continue
            text = tweet._json["full_text"]

            # Filter tweets that contain the word "giveaway"
            if "giveaway" in text.lower():
                continue
            # Filter tweets where follower count is less than 50
            if tweet.user.followers_count < 50:
                continue

                # Check if the tweet content is already in the set
            if text in tweet_set:
                continue

            # Add the tweet content to the set
            tweet_set.add(text)

            tweet_dict = {
                "user": tweet.user.screen_name,
                "text": text,
                "favorite_count": tweet.favorite_count,
                "retweet_count": tweet.retweet_count,
                "created_at": tweet.created_at
            }

            tweet_list.append(tweet_dict)
        
        # Convert the list to a DataFrame
        df = pd.DataFrame(tweet_list)

        # Use the sentiment analysis model to classify the tweets
        df = analyze_sentiment(df)

        # Filter out non-English tweets
        df = df[df["text"].apply(is_english)]

        # Filter the dataframe by sentiment label and get the counts
        filtered_df = df[df["sentiment_label"].isin(["positive", "negative", "neutral"])]
        label_counts = filtered_df["sentiment_label"].value_counts()

        # Create a bar chart for tweets sentiment
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.bar(label_counts.index, label_counts.values, color=['#00425A', '#BFDB38', '#FC7300'])
        ax1.set_title("Tweets Sentiment Analysis Results")
        ax1.set_xlabel("Sentiment Classification")
        ax1.set_ylabel("Count")
        plt.tight_layout()

        # Convert the plot to a base64 encoded image
        buffer1 = BytesIO()
        plt.savefig(buffer1, format='svg')
        buffer1.seek(0)
        encoded_plot = base64.b64encode(buffer1.getvalue()).decode()

        # Set the API endpoint and send a request to the API
        api_endpoint = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,  # search query
            "sortBy": "publishedAt",  # sort the results by publication date
            "pageSize": 20,  # get the latest 10 articles
            "apiKey": "b4a43d1ec43a49b5893c7d7cc41803c0",  # replace with your own API key
        }
        response = requests.get(api_endpoint, params=params)

        # Parse the response as JSON
        data = response.json()

        # Extract the articles from the response
        articles = data["articles"]

        # Create a list to store the article data
        article_data = []

        # Write the articles to the list
        for article in articles:
            # Get the HTML content of the article
            article_html = requests.get(article["url"]).text

            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(article_html, "html.parser")

            # Extract the data from the article
            source = article["source"]["name"]
            author = article["author"]
            title = article["title"]
            description = article["description"]
            url = article["url"]
            urlToImage = article["urlToImage"]
            publishedAt = article["publishedAt"]
            content = article["content"]

            # Add the extracted data to the article data list
            article_data.append([source, author, title, description, url, urlToImage, publishedAt, content])

        # Create a DataFrame from the article data
        df_news = pd.DataFrame(article_data, columns=["source", "author", "title", "description", "url", "urlToImage", "publishedAt", "content"])

        # Use the sentiment analysis model to classify the article descriptions
        df_news = analyze_sentiment_news(df_news)
        df_news = df_news[df_news["description"].apply(is_english)]

        # Filter the dataframe by sentiment label and get the counts
        filtered_df_news = df_news[df_news["sentiment_label"].isin(["positive", "negative", "neutral"])]
        label_counts_news = filtered_df_news["sentiment_label"].value_counts()

        # Create a bar chart for news sentiment
        fig_news, ax_news = plt.subplots(figsize=(5, 3))
        ax_news.bar(label_counts_news.index, label_counts_news.values, color=['#00425A', '#BFDB38', '#FC7300'])
        ax_news.set_title("News Sentiment Analysis Results")
        ax_news.set_xlabel("Sentiment Classification")
        ax_news.set_ylabel("Count")
        plt.tight_layout()

        # Convert the news sentiment bar chart to a base64 encoded image
        buffer_news = BytesIO()
        plt.savefig(buffer_news, format='svg')
        buffer_news.seek(0)
        encoded_news_plot = base64.b64encode(buffer_news.getvalue()).decode()

        # Render the tweets page and pass the DataFrames and encoded plots to it
        return render_template("tweets.html", data=df, news=df_news, plot=encoded_plot, news_plot=encoded_news_plot)
    else:
        return render_template("index.html")

@app.errorhandler(Exception)
def handle_error(e):
    return render_template('error.html', error=e), 500
    
if __name__ == '__main__':
    app.run(debug=True)