# %% [code] {"id":"rJZTF7UKXEMr"}
RoBERTa model

# %% [markdown] {"id":"jTYQb7wBFUrs"}
# **IMPORTS**
# 
# ---
# 
# 

# %% [code] {"id":"kxYy-5Th_oOb","executionInfo":{"status":"ok","timestamp":1690975146752,"user_tz":-330,"elapsed":14668,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"outputId":"73a83a98-1d6f-433e-8b97-7fe1d728451f","execution":{"iopub.status.busy":"2023-08-12T11:46:03.489584Z","iopub.execute_input":"2023-08-12T11:46:03.490109Z","iopub.status.idle":"2023-08-12T11:46:44.204290Z","shell.execute_reply.started":"2023-08-12T11:46:03.490063Z","shell.execute_reply":"2023-08-12T11:46:44.202966Z"}}
#scripts
!pip install emoji
!pip install transformers
!pip install gensim

# %% [code] {"id":"dXcbxd-ko02e","executionInfo":{"status":"ok","timestamp":1690975146753,"user_tz":-330,"elapsed":10,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T11:46:44.208444Z","iopub.execute_input":"2023-08-12T11:46:44.208844Z","iopub.status.idle":"2023-08-12T11:46:57.546468Z","shell.execute_reply.started":"2023-08-12T11:46:44.208808Z","shell.execute_reply":"2023-08-12T11:46:57.545325Z"}}
import pandas as pd
import tensorflow as tf
import numpy as np
import emoji as emo
import sys
import re
from transformers import BertTokenizer , BertConfig
from sklearn.preprocessing import MultiLabelBinarizer #for binary encoding of labels
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords

# %% [markdown] {"id":"RrO-L0lFusOB"}
# **DATASET AND LABEL ONE-HOT ENCODING**
# 
# ---
# 
# 

# %% [code] {"id":"FMe2OHBzsRoT","executionInfo":{"status":"ok","timestamp":1690975146754,"user_tz":-330,"elapsed":10,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"outputId":"1a360687-b2e8-461c-ebd5-2977b5b73665","execution":{"iopub.status.busy":"2023-08-12T11:46:57.548056Z","iopub.execute_input":"2023-08-12T11:46:57.548869Z","iopub.status.idle":"2023-08-12T11:46:57.743675Z","shell.execute_reply.started":"2023-08-12T11:46:57.548832Z","shell.execute_reply":"2023-08-12T11:46:57.742653Z"}}
path_train = '/kaggle/input/vaccine/val_train.csv'
path_test = '/kaggle/input/testdata/test.csv'
ds = pd.read_csv(path_train)
tds = pd.read_csv(path_test)

#converting label strings into set
lst = ds['labels'].to_list()
labels = [[label] for label in lst]
def sep(target):
    return target[0].split()

seplabels = [sep(label) for label in labels]
#one-hot encoding
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(seplabels)
label_array = np.array(labels)
tds.head()

# %% [markdown] {"id":"Gnme7J1cJGmD"}
# ## **PRE PROCESSING**
# 
# ---
# 
# 
# 

# %% [code] {"id":"X3V_8SvTH7mw","executionInfo":{"status":"ok","timestamp":1690975149955,"user_tz":-330,"elapsed":3208,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T11:46:57.747668Z","iopub.execute_input":"2023-08-12T11:46:57.749845Z","iopub.status.idle":"2023-08-12T11:47:02.191947Z","shell.execute_reply.started":"2023-08-12T11:46:57.749817Z","shell.execute_reply":"2023-08-12T11:47:02.190909Z"}}
def proc(tweet):
    tweet = tweet.lower()
    tweet = emo.demojize(tweet) #emoji to string
    tweet = re.sub(r"http[s]?://t.co/[a-zA-Z0-9]+" , "" , tweet)  #https twitter link removal
    tweet = re.sub(r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]" , "" , tweet) #unwanted expressions
    return(tweet)

tweets = ds['tweet'].apply(proc).tolist() #training data list
tests = tds['tweet'].apply(proc).tolist() #test data list






# %% [code] {"id":"50NRf12vL0fS","executionInfo":{"status":"ok","timestamp":1690975149955,"user_tz":-330,"elapsed":4,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"outputId":"ca82f2d1-98dd-4878-fc15-e7495f037bf5","execution":{"iopub.status.busy":"2023-08-12T11:47:02.193289Z","iopub.execute_input":"2023-08-12T11:47:02.193671Z","iopub.status.idle":"2023-08-12T11:47:02.202835Z","shell.execute_reply.started":"2023-08-12T11:47:02.193636Z","shell.execute_reply":"2023-08-12T11:47:02.201737Z"}}
num_classes = len(mlb.classes_)
num_classes

# %% [code] {"id":"fhNfU0uvaqsy","executionInfo":{"status":"ok","timestamp":1690975156314,"user_tz":-330,"elapsed":6362,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T11:47:02.204535Z","iopub.execute_input":"2023-08-12T11:47:02.205282Z","iopub.status.idle":"2023-08-12T11:47:24.630718Z","shell.execute_reply.started":"2023-08-12T11:47:02.205244Z","shell.execute_reply":"2023-08-12T11:47:24.629665Z"}}

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



# %% [markdown] {"id":"NXiI6RceGLry"}
# **TRAIN_TEST_SPLIT AND CONVERTING TO DICTIONARY WITH INPUTS**
# 
# ---
# 
# 

# %% [code] {"id":"O7Gu_ax3CmUo","executionInfo":{"status":"ok","timestamp":1690975172022,"user_tz":-330,"elapsed":15711,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"outputId":"2a69d04a-d996-4e95-ccdf-86f8dfccba8a","execution":{"iopub.status.busy":"2023-08-12T11:47:24.632114Z","iopub.execute_input":"2023-08-12T11:47:24.633065Z","iopub.status.idle":"2023-08-12T11:47:33.740106Z","shell.execute_reply.started":"2023-08-12T11:47:24.633028Z","shell.execute_reply":"2023-08-12T11:47:33.738932Z"}}
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
val_labels.shape

# %% [code] {"id":"HhUh4dP_S8_4","executionInfo":{"status":"ok","timestamp":1690975172023,"user_tz":-330,"elapsed":5,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T11:47:33.741810Z","iopub.execute_input":"2023-08-12T11:47:33.742195Z","iopub.status.idle":"2023-08-12T11:47:33.750826Z","shell.execute_reply.started":"2023-08-12T11:47:33.742161Z","shell.execute_reply":"2023-08-12T11:47:33.749791Z"}}
tf.keras.backend.clear_session()

# %% [markdown] {"id":"kNgtIQIaUTGu"}
# **MODEL**
# 
# ---
# 
# 

# %% [code] {"id":"hBmeLnhnxk4T","executionInfo":{"status":"ok","timestamp":1690975184111,"user_tz":-330,"elapsed":12092,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T11:47:33.752696Z","iopub.execute_input":"2023-08-12T11:47:33.753351Z","iopub.status.idle":"2023-08-12T11:47:44.236494Z","shell.execute_reply.started":"2023-08-12T11:47:33.753316Z","shell.execute_reply":"2023-08-12T11:47:44.235421Z"}}
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)
max_len = 150

# %% [code] {"id":"bTEc8IrYaEdO","executionInfo":{"status":"ok","timestamp":1690975184111,"user_tz":-330,"elapsed":10,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T11:47:44.240568Z","iopub.execute_input":"2023-08-12T11:47:44.241104Z","iopub.status.idle":"2023-08-12T11:47:44.251079Z","shell.execute_reply.started":"2023-08-12T11:47:44.241063Z","shell.execute_reply":"2023-08-12T11:47:44.249775Z"}}
tf.keras.backend.clear_session()

# %% [code] {"id":"xkzAvN3ICPcX","executionInfo":{"status":"ok","timestamp":1690975193728,"user_tz":-330,"elapsed":9626,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"outputId":"6feed225-4654-4485-ab3c-da9269ca5c5d","execution":{"iopub.status.busy":"2023-08-12T11:47:44.253315Z","iopub.execute_input":"2023-08-12T11:47:44.253733Z","iopub.status.idle":"2023-08-12T11:47:45.170777Z","shell.execute_reply.started":"2023-08-12T11:47:44.253698Z","shell.execute_reply":"2023-08-12T11:47:45.169912Z"}}
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

# %% [code] {"id":"8e_dzozX8q8V","executionInfo":{"status":"ok","timestamp":1690975193728,"user_tz":-330,"elapsed":4,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T11:47:45.171982Z","iopub.execute_input":"2023-08-12T11:47:45.172495Z","iopub.status.idle":"2023-08-12T11:47:45.182691Z","shell.execute_reply.started":"2023-08-12T11:47:45.172456Z","shell.execute_reply":"2023-08-12T11:47:45.181523Z"}}
tf.keras.backend.clear_session()

# %% [markdown] {"id":"2jzwPdDJXZOG"}
# **TRAINING**
# 
# ---
# 
# 

# %% [code] {"id":"KA3Fk33cKKQb","executionInfo":{"status":"error","timestamp":1690975860770,"user_tz":-330,"elapsed":667045,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"outputId":"385ec91c-a24e-4413-a896-9831381d444b","execution":{"iopub.status.busy":"2023-08-12T11:47:45.184328Z","iopub.execute_input":"2023-08-12T11:47:45.185805Z","iopub.status.idle":"2023-08-12T14:35:31.952052Z","shell.execute_reply.started":"2023-08-12T11:47:45.185604Z","shell.execute_reply":"2023-08-12T14:35:31.950961Z"}}
model.fit(train_x , train_labels , validation_data=(val_x , val_labels) , epochs=3, batch_size=16)
#clear GPU memory
tf.keras.backend.clear_session()

# %% [code] {"id":"F1XpLrqBRrDg","executionInfo":{"status":"aborted","timestamp":1690975860771,"user_tz":-330,"elapsed":9,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T14:35:31.956546Z","iopub.execute_input":"2023-08-12T14:35:31.956934Z","iopub.status.idle":"2023-08-12T14:35:43.473725Z","shell.execute_reply.started":"2023-08-12T14:35:31.956902Z","shell.execute_reply":"2023-08-12T14:35:43.472626Z"}}
output = model.predict(x_test)


# %% [markdown] {"id":"45krjuWFbvoO"}
# output/metrics

# %% [code] {"id":"m9UCe-a0XnAq","executionInfo":{"status":"aborted","timestamp":1690975860771,"user_tz":-330,"elapsed":9,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T14:35:43.477086Z","iopub.execute_input":"2023-08-12T14:35:43.477430Z","iopub.status.idle":"2023-08-12T14:35:43.489270Z","shell.execute_reply.started":"2023-08-12T14:35:43.477402Z","shell.execute_reply":"2023-08-12T14:35:43.488247Z"}}
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



# %% [markdown] {"id":"Ri4PepmuuiTr"}
# **MAKING CSV**
# 
# ---
# 
# 

# %% [code] {"id":"juKl1bg4ukGI","executionInfo":{"status":"aborted","timestamp":1690975860771,"user_tz":-330,"elapsed":8,"user":{"displayName":"Sohan Choudhury","userId":"03384336541033651196"}},"execution":{"iopub.status.busy":"2023-08-12T14:35:43.491766Z","iopub.execute_input":"2023-08-12T14:35:43.492189Z","iopub.status.idle":"2023-08-12T14:35:43.515722Z","shell.execute_reply.started":"2023-08-12T14:35:43.492154Z","shell.execute_reply":"2023-08-12T14:35:43.514506Z"}}
final_dict = {'id':[] , 'preds':[]}

for i in range(len(ids)):
    final_dict['id'].append(ids[i])
    tuples = pred_list[i]
    final_dict['preds'].append(' '.join(tuples))


print(final_dict)

final = pd.DataFrame(final_dict)
final.to_csv('output_11823_1e6_30.csv' , index=False)
