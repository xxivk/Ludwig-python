import logging
import pandas as pd
from ludwig.api import LudwigModel
from ludwig.visualize import learning_curves, compare_performance, compare_classifiers_predictions
from ludwig.utils.data_utils import load_json
from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text
from ludwig.utils import data_utils


# Define the configuration
config = {
    'input_features': [{
        'name': 'text',
        'type': 'text',
        'level': 'word',
        'encoder': 'parallel_cnn',
        'pretrained_embeddings': 'content/glove.6B.300d.txt',
        'embedding_size': 300
    }],
    'output_features': [{
        'name': 'label',
        'type': 'category'
    }],
    'training': {
        'decay': True,
        'learning_rate': 0.001,
        'validation_field': 'label',
        'validation_metric': 'accuracy'
    }
}

# Load the training data
train_dataset = {
    'text': ['This is a positive sentence.', 'This is a negative sentence.', 'This is another positive sentence.'],
    'label': ['positive', 'negative', 'positive']
}

# Create a Ludwig model
model = LudwigModel(config, logging_level=logging.DEBUG)
#train_stats_parallel_cnn = data_utils.load_json('train_stats_parallel_cnn.json')

# Train the model
train_stats, _, _ = model.train(
    training_set=train_dataset,
    model_name='parallel_cnn',
    skip_save_processed_input=True
)

# Access the relevant metrics

# visualizing the training results  
learning_curves(train_stats, output_feature_name='label', model_names='parallel_cnn')
