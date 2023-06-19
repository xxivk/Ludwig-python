#
#
#ludwig train --dataset ratings_sample.csv --config config.yml#
#ludwig visualize --visualization learning_curves --training_statistics ratings_sample.meta.json
#import json
#
## Load the metadata from the file
#with open('ratings_sample.meta.json', 'r') as f:
#    metadata = json.load(f)
#
## Print the metadata
#print(json.dumps(metadata, indent=4))
#ludwig predict --model_path results/experiment_run_7/model --dataset ratings_sample.csv --output_directory output
#

import pandas as pd

# Load the prediction CSV file
predictions = pd.read_csv('output/rating_predictions.csv')
#
## Access movie IDs and predicted ratings
movie_ids = predictions['movie_id']
predicted_ratings = predictions['rating']
#
# Print the results
for movie_id, rating in zip(movie_ids, predicted_ratings):
    print(f"Movie ID: {movie_id}, Predicted Rating: {rating}")
#

