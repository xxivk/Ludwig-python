import ludwig
from ludwig.api import LudwigModel
import pandas as pd
from ludwig.api import LudwigModel
from ludwig.visualize import learning_curves, compare_performance, compare_classifiers_predictions
from ludwig.utils.data_utils import load_json
from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text
from ludwig.utils import data_utils

import matplotlib.pyplot as plt

# Load the pre-trained Ludwig model
model = LudwigModel.load('.data\sst\trees\train.txt')

# Define a function for sentiment analysis using Ludwig
def analyze_sentiment(text):
    # Create a dataframe with a 'text' column
    df = pd.DataFrame({'text': [text]})

    # Use the pre-trained model to predict sentiment labels
    predictions = model.predict(data_df=df)

    # Get the predicted sentiment label
    sentiment_label = predictions['label_predictions'].values[0]

    # Map the Ludwig label to negative (0), neutral (1), and positive (2)
    sentiment_mapping = {'very negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very positive': 4}
    sentiment = sentiment_mapping[sentiment_label]

    return sentiment

# Example usage
text = "I really enjoyed the movie, it was fantastic!"
sentiment = analyze_sentiment(text)
print("Sentiment:", sentiment)

# Create a function to plot the sentiment distribution
def plot_sentiment_distribution(sentiments):
    sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    sentiment_counts = [sentiments.count(i) for i in range(5)]

    plt.bar(sentiment_labels, sentiment_counts)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    plt.show()

# Generate example text for sentiment distribution analysis
text_samples = ["This movie was terrible!", "The weather is just okay.", "I had a great time at the party!", "The food was amazing!"]

# Analyze sentiments and store the results
sentiments = []
for sample in text_samples:
    sentiment = analyze_sentiment(sample)
    sentiments.append(sentiment)

# Plot the sentiment distribution
plot_sentiment_distribution(sentiments)
