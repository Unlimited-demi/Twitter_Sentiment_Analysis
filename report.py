import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Read the analyzed validation set
val_df = pd.read_csv("analyzed_tweets_val.csv")

# Read the original validation set
val_actual_df = pd.read_csv("twitter_validation.csv")

# Remove the sentiment column from the original validation set
val_actual_df = val_actual_df.iloc[:, :2]

# Calculate the accuracy for the validation set
val_predicted_labels = val_df["SentimentScore"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
val_actual_labels = val_actual_df["sentiment"]
val_accuracy = accuracy_score(val_actual_labels, val_predicted_labels)
print("Validation Set Accuracy:", val_accuracy)

# Calculate the F1 score for the validation set
val_f1_score = f1_score(val_actual_labels, val_predicted_labels, average="weighted")
print("Validation Set F1 Score:", val_f1_score)

# Generate classification report for the validation set
classification_rep = classification_report(val_actual_labels, val_predicted_labels)
print("Classification Report:")
print(classification_rep)
