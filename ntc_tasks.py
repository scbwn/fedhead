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
    for j in ['uc75', 'uc141']:
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

def model_nn(d, num_of_classes):
    set_seed_TF2(100)
    inp=Input(shape=(d,))
    x=Dense(200, activation='relu')(inp)
    x=Dense(200, activation='relu')(x)
    x=Dense(200, activation='relu')(x)
    x=Dense(200, activation='relu')(x)
    op=Dense(num_of_classes, activation='softmax')(x)
    nn=Model(inputs=inp, outputs=op)
    #nn.summary()
    return nn

###############################################################################
# DATA PREPROCESSING
###############################################################################
with tf.device("CPU"):
    # Dataset Preprocessing
    if dataset=='uc75':
        d=81
        num_of_classes=75
        #x_train=np.load("/home/chowd531/venv/env/Unicauca/75/X_train_cos.npy")
        #y_train=np.load("/home/chowd531/venv/env/Unicauca/75/y_train_cos.npy")
        #x_test=np.load("/home/chowd531/venv/env/Unicauca/75/X_test_cos.npy")
        #y_test=np.load("/home/chowd531/venv/env/Unicauca/75/y_test_cos.npy")

        df=pd.read_csv("/home/chowd531/venv/env/Unicauca/75/app_data.csv")
        df.dropna(inplace=True)
        labels = df['L7Protocol'].iloc[:].values
        x = df.drop(columns=['L7Protocol']).iloc[:].values
        le=LabelEncoder()
        y=le.fit_transform(labels)
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)


    elif dataset=='uc141':
        d=45
        num_of_classes=141
        df=pd.read_csv("/home/chowd531/venv/env/Unicauca/141/app_data.csv")
        df.dropna(inplace=True)
        labels = df['web_service'].iloc[:].values
        x = df.drop(columns=['web_service']).iloc[:].values
        le=LabelEncoder()
        y=le.fit_transform(labels)
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)


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

    # Scaling
    scl=StandardScaler()
    x_train=scl.fit_transform(x_train)
    x_test=scl.transform(x_test)
    x_ref=scl.transform(x_ref)
    x_val=scl.transform(x_val)

    y_train=y_train.reshape(-1,1)
    y_val=y_val.reshape(-1,1)
    y_test=y_test.reshape(-1,1)

    # One-hot Encoding
    enc = OneHotEncoder(sparse=False)
    enc.fit(y.reshape(-1,1))    
    y_train=enc.transform(y_train)
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
    model_choice=model_nn(d, num_of_classes)

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