# Twitter Sentiment Analysis Project

This project focuses on performing sentiment analysis on Twitter data using the NLTK Vader module. It involves analyzing a dataset of tweets, visualizing sentiment distributions, generating word clouds, and evaluating the accuracy and F1 score of the sentiment analysis model.

## Project Structure

The project consists of the following files:

- `analyze_tweets.py`: Python script for analyzing the tweets and generating visualizations.
- `report_generation.py`: Python script for generating a report with accuracy, F1 score, and classification report.
- `nltk.py`: Python script for downloading the NLTK Vader lexicon.
- `twitter_training.csv`: CSV file containing the training set of tweets.
- `twitter_validation.csv`: CSV file containing the validation set of tweets.

## Getting Started

To get started with the project, follow these steps:

1. Ensure you have Python 3.x installed on your system.
2. Install the required dependencies by running the following command in your terminal or command prompt:
   ```
   pip install pandas matplotlib wordcloud nltk
   ```
3. Place the `analyze_tweets.py`, `report_generation.py`, `nltk.py`, `twitter_training.csv`, and `twitter_validation.csv` files in the same directory.
4. Open a terminal or command prompt, navigate to the directory containing the files, and run the following command to download the NLTK Vader lexicon:
   ```
   python nltk.py
   ```
5. After the lexicon is downloaded, you can proceed to run the analysis script by executing the following command:
   ```
   python analyze_tweets.py
   ```
6. Once the analysis is complete, you can run the report generation script to generate a report with accuracy, F1 score, and classification report:
   ```
   python report_generation.py
   ```

## Results

The project will generate visualizations of the sentiment distributions in the training and validation sets. It will also generate word clouds of the most frequently used words in the tweets. Additionally, the report generation script will provide the accuracy, F1 score, and classification report for the sentiment analysis model on the validation set.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

Feel free to modify and use the code according to your requirements.

## Acknowledgments

- The NLTK library for providing the Vader sentiment analysis module.
- The Python programming language for its simplicity and versatility.
- The pandas, matplotlib, and wordcloud libraries for data manipulation, visualization, and word cloud generation.
- The Twitter API for providing access to tweet data.

