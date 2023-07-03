import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Preprocessing
def preprocess_dataset(file_path):
    # Read the dataset from the file
    df = pd.read_csv(file_path)

    # Remove the first and second columns
    df = df.iloc[:, 2:]

    # Rename the columns to 'tweets' and 'sentiment'
    df.columns = ["tweets", "sentiment"]

    # Save the preprocessed dataset to a new file
    preprocessed_file_path = file_path.split(".csv")[0] + "_preprocessed.csv"
    df.to_csv(preprocessed_file_path, index=False)

    return preprocessed_file_path

# Read tweets from the training and validation files
train_file_path = "twitter_training.csv"
val_file_path = "twitter_validation.csv"
train_preprocessed_file_path = preprocess_dataset(train_file_path)
val_preprocessed_file_path = preprocess_dataset(val_file_path)

# Perform sentiment analysis on the training set
sia = SentimentIntensityAnalyzer()
train_df = pd.read_csv(train_preprocessed_file_path)
train_df["SentimentScore"] = train_df["tweets"].apply(lambda x: sia.polarity_scores(x)["compound"])
train_df["SentimentLabel"] = train_df["SentimentScore"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")

# Perform sentiment analysis on the validation set
val_df = pd.read_csv(val_preprocessed_file_path)
val_df["SentimentScore"] = val_df["tweets"].apply(lambda x: sia.polarity_scores(x)["compound"])

# Save the analyzed training and validation sets to CSV files (optional)
train_df.to_csv("analyzed_tweets_train.csv", index=False)
val_df.to_csv("analyzed_tweets_val.csv", index=False)

# Visualize the sentiment distribution in the training set
train_sentiment_counts = train_df["SentimentLabel"].value_counts()
plt.bar(train_sentiment_counts.index, train_sentiment_counts.values)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Analysis of Training Set Tweets")
plt.show()

# Visualize the sentiment distribution in the validation set
val_sentiment_counts = val_df["SentimentScore"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral").value_counts()
plt.bar(val_sentiment_counts.index, val_sentiment_counts.values)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Analysis of Validation Set Tweets")
plt.show()

# Generate a word cloud of the most frequently used words in the training set
train_all_tweets = " ".join(tweet for tweet in train_df["tweets"])
train_wordcloud = WordCloud(width=800, height=400).generate(train_all_tweets)
plt.figure(figsize=(10, 5))
plt.imshow(train_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of AI-Related Tweets (Training Set)")
plt.show()

# Generate a word cloud of the most frequently used words in the validation set
val_all_tweets = " ".join(tweet for tweet in val_df["tweets"])
val_wordcloud = WordCloud(width=800, height=400).generate(val_all_tweets)
plt.figure(figsize=(10, 5))
plt.imshow(val_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of AI-Related Tweets (Validation Set)")
plt.show()
