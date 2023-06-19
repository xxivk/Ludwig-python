import ludwig                                               
from ludwig.api import LudwigModel  
from ludwig.visualize import learning_curves, compare_performance, compare_classifiers_predictions
from ludwig.utils.data_utils import load_json
from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text

# data and visualization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# dataset utils
from torchtext import data                                  
from torchtext import datasets   
from torchtext.datasets import SST

# auxiliary packages
import os
import yaml                                       
import logging                                    
import json
from tqdm.notebook import tqdm                           




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

text_field = data.Field(sequential=False)
label_field = data.Field(sequential=False)  # False means no tokenization


###!###########################################################################

# obtain pre-split data into training, validation and testing sets
train_split, val_split, test_split = SST.splits(
    text_field,
    label_field,
    fine_grained=fine_grained,
    train_subtrees=True  # use all subtrees in the training set
)
# obtain texts and labels from the training set
print("Starting train loop...")
x_train = []
y_train = []
for i in tqdm(range(len(train_split)), desc='Train'):
    x_train.append(vars(train_split[i])["text"])
    y_train.append(class2idx[vars(train_split[i])["label"]])
print("Train loop finished.")

print("Starting validation loop...")
x_val = []
y_val = []
for i in tqdm(range(len(val_split)), desc='Validation'):
    x_val.append(vars(val_split[i])["text"])
    y_val.append(class2idx[vars(val_split[i])["label"]])
print("Validation loop finished.")

print("Starting test loop...")
x_test = []
y_test = []
for i in tqdm(range(len(test_split)), desc='Test'):
    x_test.append(vars(test_split[i])["text"])
    y_test.append(class2idx[vars(test_split[i])["label"]])
print("Test loop finished.")



# create three separate dataframes
train_data = pd.DataFrame({"text": x_train, "label": y_train})
validation_data = pd.DataFrame({"text": x_val, "label": y_val})
test_data = pd.DataFrame({"text": x_test, "label": y_test})


# preview sample of data with class labels
labels = [idx2class[int(id)] for id in train_data['label']]

train_data_preview = train_data.drop(columns='label').assign(class_id=train_data['label'], class_label=labels)

train_data_preview.head()
