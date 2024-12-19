import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(1)
rn.seed(2)
tf.random.set_seed(3)
import os


import pandas as pd
import keras_nlp
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence
from tensorflow.keras.layers import Embedding, LSTM, Conv2D, Conv1D, MaxPooling1D, Dense, Dropout, GlobalMaxPooling1D, Input, Bidirectional, concatenate, Flatten, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from copy import deepcopy
import time
import gc
import sys

N=20
M=2
l_round=20
s_round=20
g_round=20
comm_round=50
alpha=1#sys.argv[1]

task_id=int(sys.argv[1])
c=0
for i in ['fedavg', 'fedhead', 'feddf', 'fedhead+']:
    for j in ['sst2', 'agnews']:
        if c==task_id:
            alg=i
            dataset=j
        c+=1

print(alg)
print(dataset)

###############################################################################
# FUNCTION DEFINITION
###############################################################################
def set_seed_TF2(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    
def split_data(x_train, y_train, N, num_of_classes, s):
    client_y_list=[]
    for i in range(num_of_classes):
        x_cls=x_train[y_train[:,i]==1]
        y_cls=y_train[y_train[:,i]==1]
        chunk_list = np.floor(s[i]*np.sum(y_train[:,i])).astype(int)
        x_list=tf.split(x_cls[:np.sum(chunk_list)], chunk_list, axis=0)
        y_list=tf.split(y_cls[:np.sum(chunk_list)], chunk_list, axis=0)
        if i==0:
            client_x_list=[]
            client_y_list=[]
            for j in range(N):
                client_x_list.append(x_list[j])
                client_y_list.append(y_list[j])
        else:
            for j in range(N):
                client_x_list[j]=np.vstack([client_x_list[j], x_list[j]])
                client_y_list[j]=np.vstack([client_y_list[j], y_list[j]])

    return client_x_list, client_y_list

def model_nn():
    set_seed_TF2(100)
    model=Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=maxlen))
    model.add(Bidirectional(LSTM(128, return_sequences=True))) 
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(GlobalMaxPooling1D()) #Pooling Layer decreases sensitivity to features, thereby creating more generalised data for better test results.
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(num_of_classes, activation='softmax')) #softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.
    model.summary()
    return model

###############################################################################
# DATA PREPROCESSING
###############################################################################
with tf.device("CPU"):
    # Dataset Preprocessing
    if dataset=='agnews':
        num_of_classes=4
        data = pd.read_csv("/home/chowd531/venv/env/AG News/train.csv").dropna()
        testdata = pd.read_csv("/home/chowd531/venv/env/AG News/test.csv").dropna()

        x_train = data['Title'] + " " + data['Description']
        y_train = data['Class Index'].apply(lambda x: x-1).values # Classes need to begin from 0

        x_test = testdata['Title'] + " " + testdata['Description']
        y_test = testdata['Class Index'].apply(lambda x: x-1).values # Classes need to begin from 0

    elif dataset=='sst2':
        num_of_classes=2
        from sklearn.model_selection import train_test_split
        data = pd.read_parquet("/home/chowd531/venv/env/SST-2/train.parquet")
        x_train, x_test, y_train, y_test = train_test_split(data['sentence'], data['label'].values, test_size=0.1, random_state=100)


    maxlen = 128
    vocab_size = 10000 # arbitrarily chosen
    embed_size = 32 # arbitrarily chosen

    # Create and Fit tokenizer
    tok = Tokenizer(num_words=vocab_size)
    tok.fit_on_texts(x_train.values)

    # Tokenize data
    x_train = tok.texts_to_sequences(x_train)
    x_test = tok.texts_to_sequences(x_test)

    # Pad data
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)


    ref_size=5000#int(x_train.shape[0]/N)
    x_ref=x_train[:ref_size]
    y_ref=y_train[:ref_size]
    x_train=x_train[ref_size:]
    y_train=y_train[ref_size:]

    #validation split
    x_val = x_train[:1000]
    y_val = y_train[:1000]
    x_train = x_train[1000:]
    y_train = y_train[1000:]


    y_train=y_train.reshape(-1,1)
    y_val=y_val.reshape(-1,1)
    y_test=y_test.reshape(-1,1)

    # One-hot Encoding
    enc = OneHotEncoder(sparse=False)
    y_train=enc.fit_transform(y_train)
    y_val=enc.transform(y_val)
    y_train=y_train.astype(np.float32)
    y_val=y_val.astype(np.float32)


    alpha_list = [alpha] * N
    np.random.seed(1)
    s = np.random.dirichlet(alpha_list, num_of_classes)
    x_train_list, y_train_list = split_data(x_train, y_train, N, num_of_classes, s)

    D_mat=np.zeros([N, num_of_classes])
    for n in range(N):
        D_mat[n]=np.sum(y_train_list[n], axis=0)
    model_choice=model_nn()

###############################################################################
# FL ALGORITHM
###############################################################################
if alg=='fedavg':
    from fedlib_new import FedAvg
    alg_choice=FedAvg(N, l_round, comm_round, model_choice)
    global_model=alg_choice.train_model(D_mat, x_train_list, y_train_list, x_test, y_test)
if alg=='fedhead':
    from fedlib_new import FedHEAD
    alg_choice=FedHEAD(N, M, l_round, s_round, comm_round, model_choice)
    global_model=alg_choice.train_model(D_mat, x_train_list, y_train_list, x_test, y_test, x_val, y_val)
if alg=='feddf':
    from fedlib_new import FedDF
    alg_choice=FedDF(N, l_round, g_round, comm_round, model_choice)
    global_model=alg_choice.train_model(D_mat, x_train_list, y_train_list, x_ref, x_test, y_test, x_val, y_val)
if alg=='fedhead+':
    from fedlib_new import FedHEAD_plus
    alg_choice=FedHEAD_plus(N, M, l_round, s_round, g_round, comm_round, model_choice)
    global_model=alg_choice.train_model(D_mat, x_train_list, y_train_list, x_ref, x_test, y_test, x_val, y_val)