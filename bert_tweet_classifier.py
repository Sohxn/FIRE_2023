
#RoBERTa model


#DEPENDENCIES
#!pip install emoji
#!pip install transformers


import pandas as pd
import tensorflow as tf
import numpy as np
import emoji as emo
import re
from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer 
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub



path_train = 'datasets/train.csv'
path_test = 'datasets/test.csv'
ds = pd.read_csv(path_train)
tds = pd.read_csv(path_test)

lst = ds['labels'].to_list()
labels = [[label] for label in lst]
def sep(target):
    return target[0].split()

seplabels = [sep(label) for label in labels]

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(seplabels)
label_array = np.array(labels)

 

def proc(tweet):
    tweet = tweet.lower()
    tweet = emo.demojize(tweet) #emoji to string
    tweet = re.sub(r"http[s]?://t.co/[a-zA-Z0-9]+" , "" , tweet)  #https twitter link removal
    tweet = re.sub(r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]" , "" , tweet) #unwanted expressions
    return(tweet)

tweets = ds['tweet'].apply(proc).tolist() #training data list
tests = tds['tweet'].apply(proc).tolist() #test data list






num_classes = len(mlb.classes_)

tok = BertTokenizer.from_pretrained('bert-base-uncased')
tweet_encoded_training = tok(tweets , padding='max_length' , truncation=True , max_length = 150 , return_tensors='tf')
tweet_encoded_test = tok(tests , padding='max_length', truncation = True , max_length = 150 , return_tensors='tf')

x_train = {
    'input_word_ids':tweet_encoded_training['input_ids'],
    'input_mask': tweet_encoded_training['attention_mask'],
    'input_type_ids': tweet_encoded_training['token_type_ids']
}

x_test = {
    'input_word_ids': tweet_encoded_test['input_ids'],
    'input_mask': tweet_encoded_test['attention_mask'],
    'input_type_ids': tweet_encoded_test['token_type_ids']
}

data_list = [({'input_word_ids': x_train['input_word_ids'][i],
               'input_mask': x_train['input_mask'][i],
               'segment_ids': x_train['input_type_ids'][i]}, label) for i, label in enumerate(label_array)]

train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42)

train_x = {'input_word_ids': np.array([elem[0]['input_word_ids'] for elem in train_list]),
           'input_mask': np.array([elem[0]['input_mask'] for elem in train_list]),
           'input_type_ids': np.array([elem[0]['segment_ids'] for elem in train_list])}

val_x = {'input_word_ids': np.array([elem[0]['input_word_ids'] for elem in val_list]),
         'input_mask': np.array([elem[0]['input_mask'] for elem in val_list]),
         'input_type_ids': np.array([elem[0]['segment_ids'] for elem in val_list])}

train_labels = np.array([elem[1] for elem in train_list])
val_labels = np.array([elem[1] for elem in val_list])

tf.keras.backend.clear_session()

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)
max_len = 150

tf.keras.backend.clear_session()

def build_model(num_classes):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output , sequence_output = bert_layer([input_word_ids , input_mask , segment_ids])
    dense = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
    drop = tf.keras.layers.Dropout(0.1)(dense)
    dense2 = tf.keras.layers.Dense(64 , activation='relu')(drop)
    drop2 = tf.keras.layers.Dropout(0.1)(dense2)
    output = tf.keras.layers.Dense(num_classes , activation = 'sigmoid' , name = 'output')(drop2)

    model = tf.keras.Model(inputs = {
        'input_word_ids' : input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    }, outputs = output)

    return model

num_classes = 12
model = build_model(num_classes)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = 'accuracy')

model.summary()

tf.keras.backend.clear_session()



model.fit(train_x , train_labels , validation_data=(val_x , val_labels) , epochs=3, batch_size=16)

tf.keras.backend.clear_session()

output = model.predict(x_test)


pred = (output > 0.25).astype(int)

all_labels = mlb.classes_
pred_list = mlb.inverse_transform(pred)
ids = tds['id']
#for choosing most likely labels(countering empty tuples)
for i , tup in enumerate(pred_list):
    if not tup: #if the tuple is empty
      #analysing probs
      prob = output[i]
      top_index = prob.argsort()[-3:] #top 3 labels
      final_label = all_labels[top_index]
      pred_list[i] = tuple(final_label)



final_dict = {'id':[] , 'preds':[]}

for i in range(len(ids)):
    final_dict['id'].append(ids[i])
    tuples = pred_list[i]
    final_dict['preds'].append(' '.join(tuples))


print(final_dict)

final = pd.DataFrame(final_dict)
final.to_csv('output_11823_1e6_30.csv' , index=False)
