import ludwig                                               
from ludwig.api import LudwigModel  
from ludwig.visualize import learning_curves, compare_performance, compare_classifiers_predictions
from ludwig.utils.data_utils import load_json
from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text

# data and visualization
import spacy
from spacy.pipeline.tagger import Tagger



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# dataset utils
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torchtext import datasets   

# auxiliary packages
import os
import yaml                                       
import logging                                    
import json
from tqdm.notebook import trange   





# pick either SST-2 (False) or SST-5 (True)
fine_grained = True

if(fine_grained):
  # define SST-5 classes for sentiment labels
  idx2class = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
  class2idx = {cls: idx for idx, cls in enumerate(idx2class)}
else:
  # define SST-2 classes for sentiment labels
  idx2class = ["negative", "neutral", "positive"]
  class2idx = {cls: idx for idx, cls in enumerate(idx2class)}

text_field = Field(sequential=False)
label_field = Field(sequential=True)  # False means no tokenization

# obtain pre-split data into training, validation and testing sets
train_split, val_split, test_split = datasets.SST.splits(
    text_field,
    label_field,
    fine_grained=fine_grained,
    train_subtrees=True  # use all subtrees in the training set
)

# obtain texts and labels from the training set
x_train = []
y_train = []
for i in trange(len(train_split), desc='Train'):
    x_train.append(vars(train_split[i])["text"])
    y_train.append(class2idx[train_split[i].label[0]])

    #y_train.append(class2idx[vars(train_split[i])["label"]])

# obtain texts and labels from the validation set
x_val = []
y_val = []
for i in trange(len(val_split), desc='Validation'):
    x_val.append(vars(val_split[i])["text"])
    #y_val.append(class2idx[vars(val_split[i])["label"]])
    y_val.append(class2idx[val_split[i].label[0]])


# obtain texts and labels from the test set
x_test = []
y_test = []
for i in trange(len(test_split), desc='Test'):
    x_test.append(vars(test_split[i])["text"])
    #y_test.append(class2idx[vars(test_split[i])["label"]])
    y_test.append(class2idx[test_split[i].label[0]])


# create three separate dataframes
train_data = pd.DataFrame({"text": x_train, "label": y_train})
validation_data = pd.DataFrame({"text": x_val, "label": y_val})
test_data = pd.DataFrame({"text": x_test, "label": y_test})

# preview sample of data with class labels
labels = [idx2class[int(id)] for id in train_data['label']]

train_data_preview = train_data.drop(columns='label').assign(class_id=train_data['label'], class_label=labels)

train_data_preview.head()

# plotting look
plt.style.use('ggplot')

fig, ax = plt.subplots()

if fine_grained:
  ax.set_title(f'Distribution of sentiment labels in the SST-5 training set')
  ax = train_data['label'].value_counts(sort=False).plot(kind='barh', color=['red', 'coral', 'grey', 'lime', 'green'])
else:
  ax.set_title(f'Distribution of sentiment labels in the SST-2 training set')
  ax = train_data['label'].value_counts(sort=False).plot(kind='barh', color=['red','green'])

# axes info
ax.set_xlabel('Samples in the training set')
ax.set_ylabel('Labels')
ax.set_yticklabels(tuple(idx2class))
ax.grid(True)
#plt.show()

nlp = load_nlp_pipeline('en')
nlp.max_length = 7389814 
#nlp.remove_pipe('tagger')
#nlp.add_pipe('tagger', name='tagger')

#nlp.add_pipe('attribute_ruler', name='attribute_ruler', before='lemmatizer')

# Add POS tagger component to the pipeline
#tagger = Tagger(nlp.vocab,[],[])
#nlp.add_pipe(tagger, name='tagger', before='lemmatizer')

#nlp.add_pipe(nlp.create_pipe('tagger'), name='tagger', before='lemmatizer')
#nlp.add_pipe(tagger, name='tagger', before='lemmatizer')

processed_train_data = process_text(' '.join(train_data['text']),
                                    load_nlp_pipeline('en'),
                                    filter_punctuation=True,
                                    filter_stopwords=True)

wordcloud = WordCloud(background_color='black', collocations=False,
                      stopwords=STOPWORDS).generate(' '.join(processed_train_data))

plt.figure(figsize=(8,8))
plt.imshow(wordcloud.recolor(color_func=lambda *args, **kwargs:'white'), interpolation='bilinear')
plt.axis('off')
plt.show()

