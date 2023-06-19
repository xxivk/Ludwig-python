config = {
    'input_features': [{ 
        'name': 'text',
        'type': 'text', 
        'level': 'word', 
        'encoder': 'parallel_cnn',
        'pretrained_embeddings': '/content/glove/glove.6B.300d.txt',
        'embedding_size': 300,
        'preprocessing': { 'word_vocab_file': '/glove.6B.300d.txt' }
    }],
    'output_features': [{'name': 'label', 'type': 'category'}],
    'training': {
        'decay': True,
        'learning_rate': 0.001,
        'validation_field': 'label',
        'validation_metric': 'accuracy'
    }
}
