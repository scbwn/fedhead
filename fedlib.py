import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(1)
rn.seed(2)
tf.random.set_seed(3)
import os

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, clone_model
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from copy import deepcopy
import time
import gc

batch_size=128
loss_fn = CategoricalCrossentropy()
mse_loss = MeanSquaredError()
optimizer=Adam(decay=1E-4)

def custom_T(x,T=1):
    return tf.math.softmax(tf.math.log(x+1E-15)/T)

def lr_scheduler(epoch):
    lr=1E-3
    return lr

def sector_design(D,M):
    beta=100
    rho=0.3
    N = D.shape[0] 
    C = D.shape[1]
    w=tf.Variable(tf.random.normal([M,N], seed=100))
    for step in range(100):
        tf.random.set_seed(3)
        with tf.GradientTape() as tape:
            Z = tf.math.softmax(beta*w, axis=0)
            A=Z @ D
            b = A @ tf.ones([C,1])
            A_tilde = A-(1/C)*A @ tf.ones([C,C])
            b_tilde = b-(1/M)*tf.ones([1,M]) @ b
            objective = (1-rho)*tf.norm(A_tilde)**2+rho*tf.norm(b_tilde)**2
        grads = tape.gradient(objective, w)
        w.assign(w - 1E-3 * grads)
    return tf.math.round(tf.math.softmax(beta*w, axis=0)).numpy()

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def confidence_interval(a,l):
    import scipy.stats as st
    return st.t.interval(l, len(a)-1, loc=np.mean(a), scale=st.sem(a))

def bootstrap_score(y_test, y_pred, metric=accuracy_score, l=0.95, seed=100):
    rng = np.random.RandomState(seed=seed)
    idx = np.arange(y_test.shape[0])
    test_accuracies = []
    for i in range(200):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        acc_test_boot = metric(y_test[pred_idx], y_pred[pred_idx])
        test_accuracies.append(acc_test_boot)
    bootstrap_score_mean = np.mean(test_accuracies)
    [ci_lower, ci_upper] = confidence_interval(test_accuracies,l)
    return bootstrap_score_mean, 0.5*(ci_upper-ci_lower)

class FedAvg:
    def __init__(self, N, l_round, comm_round, model_choice):
        self.N=N
        self.l_round=l_round
        self.comm_round=comm_round
        self.model_choice = model_choice
    def train_model(self, D_mat, x_train_list, y_train_list, x_test, y_test):
        # Instantiate global_model and scaling weights for all clients
        scaling_factor_list=D_mat.sum(1)/D_mat.sum()
        acc_list=[]
        global_model=clone_model(self.model_choice)

        for t in range(self.comm_round):
            print("\nGlobal round %d" % (t,))
            optimizer.learning_rate=lr_scheduler(t)
            scaled_weight_list=[]
            for n in range(self.N):
                print("\nClient %d" % (n,))
                local_model=clone_model(self.model_choice)
                local_model.set_weights(global_model.get_weights())
                # Data generator for training data
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                train_generator = ImageDataGenerator()

                # Generate training batches
                train_batches = train_generator.flow(x_train_list[n], y_train_list[n], batch_size=batch_size)

                for epoch in range(self.l_round):
                    gc.collect()
                    print("\nStart of epoch %d" % (epoch,))

                    # Iterate over the batches of the dataset.
                    for step in range(len(x_train_list[n])//batch_size):
                        (x_batch, y_batch) = next(train_batches)

                        # Open a GradientTape to record the operations run
                        # during the forward pass, which enables auto-differentiation.
                        tf.random.set_seed(3)
                        with tf.GradientTape() as tape:

                            # Run the forward pass of the layer.
                            # The operations that the layer applies
                            # to its inputs are going to be recorded
                            # on the GradientTape.
                            y_pred = local_model(x_batch, training=True)  # predictions for this minibatch

                            # Compute the loss value for this minibatch.
                            loss_value = loss_fn(y_batch, y_pred)

                        # Use the gradient tape to automatically retrieve
                        # the gradients of the trainable variables with respect to the loss.
                        grads = tape.gradient(loss_value, local_model.trainable_weights)

                        # Run one step of gradient descent by updating
                        # the value of the variables to minimize the loss.
                        optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

                        # Log every 10 batches.
                        if step % 10 == 0:
                            print(
                                "Training loss (for one batch) at step %d: %.4f"
                                % (step, float(loss_value))
                            )
                            print("Seen so far: %s samples" % ((step + 1) * batch_size))

                scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor_list[n])
                scaled_weight_list.append(scaled_weights)


            print("\nServer")
            #to get the average over all the local model, we simply take the sum of the scaled weights
            average_weights = sum_scaled_weights(scaled_weight_list)

            #update global model
            global_model.set_weights(average_weights)

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            print(acc_list)
        return global_model

class FedHEAD:
    def __init__(self, N, M, l_round, s_round, comm_round, model_choice, random_leader=True):
        self.N=N
        self.M=M
        self.l_round=l_round
        self.s_round=s_round
        self.comm_round=comm_round
        self.model_choice = model_choice
        self.random_leader = random_leader
    def train_model(self, D_mat, x_train_list, y_train_list, x_test, y_test, x_val, y_val):
        # Instantiate global_model and scaling weights for all clients
        n_sec=np.zeros(self.M)
        i=1
        while np.all(n_sec)==False:
            np.random.seed(i)
            Z_mat=np.eye(self.M)[:,np.random.choice(self.M, self.N)]
            n_sec=Z_mat.sum(1)
            i+=1
        idx=[]
        for m in range(self.M):
            idx.append(list(np.where(Z_mat[m]==1)[0]))
        scaling_factor_list=D_mat.sum(1)/D_mat.sum()
        pre_sector_dist_acc_list=[]
        acc_list=[]
        global_model=clone_model(self.model_choice)
        
        for t in range(self.comm_round):
            print("\nGlobal round %d" % (t,))
            optimizer.learning_rate=lr_scheduler(t)
            if(self.random_leader):
                leader_id=[int(np.random.choice(idx[m], 1, p=scaling_factor_list[Z_mat[m]==1]/np.sum(scaling_factor_list[Z_mat[m]==1]))) for m in range(self.M)]
            else:
                leader_id=[int(idx[m][np.argmax(scaling_factor_list[Z_mat[m]==1])]) for m in range(self.M)]
            pre_weight_list=[]
            ens_pred_list=[]
            
            for m in range(self.M):
                scaled_weight_list=[]
                scaled_pred_list=[]
                for n in idx[m]:
                    print("\nClient %d in Sector %d" % (n,m))
                    local_model=clone_model(self.model_choice)
                    local_model.set_weights(global_model.get_weights())
                    # Data generator for training data
                    from tensorflow.keras.preprocessing.image import ImageDataGenerator
                    train_generator = ImageDataGenerator()

                    # Generate training batches
                    train_batches = train_generator.flow(x_train_list[n], y_train_list[n], batch_size=batch_size)


                    for epoch in range(self.l_round):
                        gc.collect()
                        print("\nStart of epoch %d" % (epoch,))

                        # Iterate over the batches of the dataset.
                        for step in range(len(x_train_list[n])//batch_size):
                            (x_batch, y_batch) = next(train_batches)

                            # Open a GradientTape to record the operations run
                            # during the forward pass, which enables auto-differentiation.
                            tf.random.set_seed(3)
                            with tf.GradientTape() as tape:

                                # Run the forward pass of the layer.
                                # The operations that the layer applies
                                # to its inputs are going to be recorded
                                # on the GradientTape.
                                y_pred = local_model(x_batch, training=True)  # predictions for this minibatch

                                # Compute the loss value for this minibatch.
                                loss_value = loss_fn(y_batch, y_pred)

                            # Use the gradient tape to automatically retrieve
                            # the gradients of the trainable variables with respect to the loss.
                            grads = tape.gradient(loss_value, local_model.trainable_weights)

                            # Run one step of gradient descent by updating
                            # the value of the variables to minimize the loss.
                            optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

                            # Log every 10 batches.
                            if step % 10 == 0:
                                print(
                                    "Training loss (for one batch) at step %d: %.4f"
                                    % (step, float(loss_value))
                                )
                                print("Seen so far: %s samples" % ((step + 1) * batch_size))

                    scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor_list[n]/np.sum(scaling_factor_list[Z_mat[m]==1]))
                    scaled_weight_list.append(scaled_weights)
                    scaled_preds = custom_T(local_model.predict(x_train_list[leader_id[m]]))*scaling_factor_list[n]/np.sum(scaling_factor_list[Z_mat[m]==1])
                    scaled_pred_list.append(scaled_preds)

                #Pre-distillation Sector Aggregation
                #to get the average over all the local model, we simply take the sum of the scaled weights
                average_weights = sum_scaled_weights(scaled_weight_list)
                average_preds = np.sum(scaled_pred_list, axis=0)
                print(average_preds.shape)

                scaled_weights = scale_model_weights(average_weights, np.sum(scaling_factor_list[Z_mat[m]==1]))
                pre_weight_list.append(scaled_weights)
                ens_pred_list.append(average_preds)

            print("\nServer")
            #Pre-distillation Server Aggregation
            average_weights = sum_scaled_weights(pre_weight_list)

            #update global model
            global_model.set_weights(average_weights)

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            pre_sector_dist_acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            #Sector Distillation
            post_weight_list=[]
            for m in range(self.M):
                print("\nClient Leader %d in Sector %d" % (leader_id[m],m))

                # Generate reference batches
                dist_batches = train_generator.flow(x_train_list[leader_id[m]], ens_pred_list[m], batch_size=batch_size)

                #update leader model
                leader_model=clone_model(self.model_choice)
                leader_model.set_weights(global_model.get_weights())
                callback = EarlyStopping(monitor='val_loss', patience=5)
                leader_model.compile(optimizer=optimizer, loss=loss_fn)
                tf.random.set_seed(3)
                leader_model.fit(dist_batches, batch_size=batch_size, epochs=self.s_round, verbose=2,
                                                    callbacks=[callback], validation_data=(x_val, y_val))

                scaled_weights = scale_model_weights(leader_model.get_weights(), np.sum(scaling_factor_list[Z_mat[m]==1]))
                post_weight_list.append(scaled_weights)

            print("\nServer")
            #Post-distillation Server Aggregation
            average_weights = sum_scaled_weights(post_weight_list)

            #update global model
            global_model.set_weights(average_weights)

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            print(pre_sector_dist_acc_list)
            print(acc_list)
        return global_model
    
class FedDF:
    def __init__(self, N, l_round, g_round, comm_round, model_choice):
        self.N=N
        self.l_round=l_round
        self.g_round=g_round
        self.comm_round=comm_round
        self.model_choice = model_choice
    def train_model(self, D_mat, x_train_list, y_train_list, x_ref, x_test, y_test, x_val, y_val):
        # Instantiate global_model and scaling weights for all clients
        scaling_factor_list=D_mat.sum(1)/D_mat.sum()
        pre_dist_acc_list=[]
        acc_list=[]
        global_model=clone_model(self.model_choice)
        
        for t in range(self.comm_round):
            print("\nGlobal round %d" % (t,))
            optimizer.learning_rate=lr_scheduler(t)
            scaled_weight_list=[]
            scaled_pred_list=[]
            for n in range(self.N):
                print("\nClient %d" % (n,))
                local_model=clone_model(self.model_choice)
                local_model.set_weights(global_model.get_weights())
                # Data generator for training data
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                train_generator = ImageDataGenerator()

                # Generate training batches
                train_batches = train_generator.flow(x_train_list[n], y_train_list[n], batch_size=batch_size)

                for epoch in range(self.l_round):
                    gc.collect()
                    print("\nStart of epoch %d" % (epoch,))

                    # Iterate over the batches of the dataset.
                    for step in range(len(x_train_list[n])//batch_size):
                        (x_batch, y_batch) = next(train_batches)

                        # Open a GradientTape to record the operations run
                        # during the forward pass, which enables auto-differentiation.
                        tf.random.set_seed(3)
                        with tf.GradientTape() as tape:

                            # Run the forward pass of the layer.
                            # The operations that the layer applies
                            # to its inputs are going to be recorded
                            # on the GradientTape.
                            y_pred = local_model(x_batch, training=True)  # predictions for this minibatch

                            # Compute the loss value for this minibatch.
                            loss_value = loss_fn(y_batch, y_pred)

                        # Use the gradient tape to automatically retrieve
                        # the gradients of the trainable variables with respect to the loss.
                        grads = tape.gradient(loss_value, local_model.trainable_weights)

                        # Run one step of gradient descent by updating
                        # the value of the variables to minimize the loss.
                        optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

                        # Log every 10 batches.
                        if step % 10 == 0:
                            print(
                                "Training loss (for one batch) at step %d: %.4f"
                                % (step, float(loss_value))
                            )
                            print("Seen so far: %s samples" % ((step + 1) * batch_size))

                scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor_list[n])
                scaled_weight_list.append(scaled_weights)
                scaled_preds = custom_T(local_model.predict(x_ref))*scaling_factor_list[n]
                scaled_pred_list.append(scaled_preds)

            print("\nServer")
            #to get the average over all the local model, we simply take the sum of the scaled weights
            average_weights = sum_scaled_weights(scaled_weight_list)
            average_preds = np.sum(scaled_pred_list, axis=0)
            
            #update global model
            global_model.set_weights(average_weights)

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            pre_dist_acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            # Generate reference batches
            dist_batches = train_generator.flow(x_ref, average_preds, batch_size=batch_size)
            
            callback = EarlyStopping(monitor='val_loss', patience=5)
            global_model.compile(optimizer=optimizer, loss=loss_fn)
            tf.random.set_seed(3)
            global_model.fit(dist_batches, batch_size=batch_size, epochs=self.g_round, verbose=2,
                                                callbacks=[callback], validation_data=(x_val, y_val))

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            print(pre_dist_acc_list)
            print(acc_list)
            
        return global_model
    
class FedHEAD_plus:
    def __init__(self, N, M, l_round, s_round, g_round, comm_round, model_choice, random_leader=True):
        self.N=N
        self.M=M
        self.l_round=l_round
        self.s_round=s_round
        self.g_round=g_round
        self.comm_round=comm_round
        self.model_choice = model_choice
        self.random_leader = random_leader
    def train_model(self, D_mat, x_train_list, y_train_list, x_ref, x_test, y_test, x_val, y_val):
        # Instantiate global_model and scaling weights for all clients
        n_sec=np.zeros(self.M)
        i=1
        while np.all(n_sec)==False:
            np.random.seed(i)
            Z_mat=np.eye(self.M)[:,np.random.choice(self.M, self.N)]
            n_sec=Z_mat.sum(1)
            i+=1
        idx=[]
        for m in range(self.M):
            idx.append(list(np.where(Z_mat[m]==1)[0]))
        scaling_factor_list=D_mat.sum(1)/D_mat.sum()
        pre_sector_dist_acc_list=[]
        pre_server_dist_acc_list=[]
        acc_list=[]
        global_model=clone_model(self.model_choice)
        
        for t in range(self.comm_round):
            print("\nGlobal round %d" % (t,))
            optimizer.learning_rate=lr_scheduler(t)
            if(self.random_leader):
                leader_id=[int(np.random.choice(idx[m], 1, p=scaling_factor_list[Z_mat[m]==1]/np.sum(scaling_factor_list[Z_mat[m]==1]))) for m in range(self.M)]
            else:
                leader_id=[int(idx[m][np.argmax(scaling_factor_list[Z_mat[m]==1])]) for m in range(self.M)]
            pre_weight_list=[]
            ens_pred_list=[]
            for m in range(self.M):
                scaled_weight_list=[]
                scaled_pred_list=[]
                scaled_leader_pred_list=[]
                for n in idx[m]:
                    print("\nClient %d in Sector %d" % (n,m))
                    local_model=clone_model(self.model_choice)
                    local_model.set_weights(global_model.get_weights())
                    # Data generator for training data
                    from tensorflow.keras.preprocessing.image import ImageDataGenerator
                    train_generator = ImageDataGenerator()

                    # Generate training batches
                    train_batches = train_generator.flow(x_train_list[n], y_train_list[n], batch_size=batch_size)


                    for epoch in range(self.l_round):
                        gc.collect()
                        print("\nStart of epoch %d" % (epoch,))

                        # Iterate over the batches of the dataset.
                        for step in range(len(x_train_list[n])//batch_size):
                            (x_batch, y_batch) = next(train_batches)

                            # Open a GradientTape to record the operations run
                            # during the forward pass, which enables auto-differentiation.
                            tf.random.set_seed(3)
                            with tf.GradientTape() as tape:

                                # Run the forward pass of the layer.
                                # The operations that the layer applies
                                # to its inputs are going to be recorded
                                # on the GradientTape.
                                y_pred = local_model(x_batch, training=True)  # predictions for this minibatch

                                # Compute the loss value for this minibatch.
                                loss_value = loss_fn(y_batch, y_pred)

                            # Use the gradient tape to automatically retrieve
                            # the gradients of the trainable variables with respect to the loss.
                            grads = tape.gradient(loss_value, local_model.trainable_weights)

                            # Run one step of gradient descent by updating
                            # the value of the variables to minimize the loss.
                            optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

                            # Log every 10 batches.
                            if step % 10 == 0:
                                print(
                                    "Training loss (for one batch) at step %d: %.4f"
                                    % (step, float(loss_value))
                                )
                                print("Seen so far: %s samples" % ((step + 1) * batch_size))

                    scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor_list[n]/np.sum(scaling_factor_list[Z_mat[m]==1]))
                    scaled_weight_list.append(scaled_weights)
                    scaled_preds = custom_T(local_model.predict(x_train_list[leader_id[m]]))*scaling_factor_list[n]/np.sum(scaling_factor_list[Z_mat[m]==1])
                    scaled_pred_list.append(scaled_preds)

                #Pre-distillation Sector Aggregation
                #to get the average over all the local model, we simply take the sum of the scaled weights
                average_weights = sum_scaled_weights(scaled_weight_list)
                average_preds = np.sum(scaled_pred_list, axis=0)
                print(average_preds.shape)

                scaled_weights = scale_model_weights(average_weights, np.sum(scaling_factor_list[Z_mat[m]==1]))
                pre_weight_list.append(scaled_weights)
                ens_pred_list.append(average_preds)

            print("\nServer")
            #Pre-distillation Server Aggregation
            average_weights = sum_scaled_weights(pre_weight_list)

            #update global model
            global_model.set_weights(average_weights)

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            pre_sector_dist_acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            #Sector Distillation
            post_weight_list=[]
            for m in range(self.M):
                print("\nClient Leader %d in Sector %d" % (leader_id[m],m))

                # Generate reference batches
                dist_batches = train_generator.flow(x_train_list[leader_id[m]], ens_pred_list[m], batch_size=batch_size)


                #update leader model
                leader_model=clone_model(self.model_choice)
                leader_model.set_weights(global_model.get_weights())
                callback = EarlyStopping(monitor='val_loss', patience=5)
                leader_model.compile(optimizer=optimizer, loss=loss_fn)
                tf.random.set_seed(3)
                leader_model.fit(dist_batches, batch_size=batch_size, epochs=self.s_round, verbose=2,
                                                    callbacks=[callback], validation_data=(x_val, y_val))

                scaled_preds = custom_T(leader_model.predict(x_ref))*np.sum(scaling_factor_list[Z_mat[m]==1])
                scaled_leader_pred_list.append(scaled_preds)
                scaled_weights = scale_model_weights(leader_model.get_weights(), np.sum(scaling_factor_list[Z_mat[m]==1]))
                post_weight_list.append(scaled_weights)
                
            print("\nServer")
            #Post-distillation Server Aggregation
            average_weights = sum_scaled_weights(post_weight_list)
            average_preds = np.sum(scaled_leader_pred_list, axis=0)
            
            #update global model
            global_model.set_weights(average_weights)
            
            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            pre_server_dist_acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            # Generate reference batches
            dist_batches = train_generator.flow(x_ref, average_preds, batch_size=batch_size)

            callback = EarlyStopping(monitor='val_loss', patience=5)
            global_model.compile(optimizer=optimizer, loss=loss_fn)
            tf.random.set_seed(3)
            global_model.fit(dist_batches, batch_size=batch_size, epochs=self.g_round, verbose=2,
                                                callbacks=[callback], validation_data=(x_val, y_val))

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            print(pre_sector_dist_acc_list)
            print(pre_server_dist_acc_list)
            print(acc_list)
        return global_model    

class FedHKD:
    def __init__(self, N, l_round, comm_round, conv_choice, clf_choice):
        self.N=N
        self.l_round=l_round
        self.comm_round=comm_round
        self.conv_choice = conv_choice
        self.clf_choice = clf_choice
    def train_model(self, D_mat, x_train_list, y_train_list, x_test, y_test):
        # Instantiate global_model and scaling weights for all clients
        scaling_factor_list=D_mat.sum(1)/D_mat.sum()
        nu_mat=np.int32(D_mat/D_mat.sum(1).reshape(-1,1) > 0.25 )
        cls_scaling_factor_list=(nu_mat*D_mat)/(nu_mat*D_mat+1E-15).sum(0)
        acc_list=[]
        global_conv=clone_model(self.conv_choice)
        global_clf=clone_model(self.clf_choice)

        for t in range(self.comm_round):
            print("\nGlobal round %d" % (t,))
            optimizer.learning_rate=lr_scheduler(t)
            scaled_conv_weight_list=[]
            scaled_clf_weight_list=[]
            scaled_clf_preds_list=[]
            scaled_conv_preds_list=[]
            for n in range(self.N):
                print("\nClient %d" % (n,))
                local_conv=clone_model(self.conv_choice)
                local_clf=clone_model(self.clf_choice)
                local_conv.set_weights(global_conv.get_weights())
                local_clf.set_weights(global_clf.get_weights())
                # Data generator for training data
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                train_generator = ImageDataGenerator()

                # Generate training batches
                train_batches = train_generator.flow(x_train_list[n], y_train_list[n], batch_size=batch_size)

                for epoch in range(self.l_round):
                    gc.collect()
                    print("\nStart of epoch %d" % (epoch,))

                    # Iterate over the batches of the dataset.
                    for step in range(len(x_train_list[n])//batch_size):
                        (x_batch, y_batch) = next(train_batches)

                        # Open a GradientTape to record the operations run
                        # during the forward pass, which enables auto-differentiation.
                        tf.random.set_seed(3)
                        with tf.GradientTape(persistent=True) as tape:

                            # Run the forward pass of the layer.
                            # The operations that the layer applies
                            # to its inputs are going to be recorded
                            # on the GradientTape.
                            y_pred = local_clf(local_conv(x_batch, training=True), training=True)  # predictions for this minibatch
                            
                            # Compute the loss value for this minibatch.
                            if t==0:
                                loss_value = loss_fn(y_batch, y_pred)
                            else:
                                loss_a=loss_fn(y_batch, y_pred)
                                loss_b=loss_fn(average_clf_preds, custom_T(local_clf(average_conv_preds, training=True), T=0.5))
                                loss_c=mse_loss(average_conv_preds[np.argmax(y_batch,1)], local_conv(x_batch, training=True))
                                loss_value = loss_a+0.5*loss_b+0.5*loss_c

                        # Use the gradient tape to automatically retrieve
                        # the gradients of the trainable variables with respect to the loss.
                        grads_conv = tape.gradient(loss_value, local_conv.trainable_weights)
                        grads_clf = tape.gradient(loss_value, local_clf.trainable_weights)

                        # Run one step of gradient descent by updating
                        # the value of the variables to minimize the loss.
                        optimizer.apply_gradients(zip(grads_conv, local_conv.trainable_weights))
                        optimizer.apply_gradients(zip(grads_clf, local_clf.trainable_weights))

                        # Log every 10 batches.
                        if step % 10 == 0:
                            print(
                                "Training loss (for one batch) at step %d: %.4f"
                                % (step, float(loss_value))
                            )
                            print("Seen so far: %s samples" % ((step + 1) * batch_size))

                scaled_weights = scale_model_weights(local_conv.get_weights(), scaling_factor_list[n])
                scaled_conv_weight_list.append(scaled_weights)
                scaled_weights = scale_model_weights(local_clf.get_weights(), scaling_factor_list[n])
                scaled_clf_weight_list.append(scaled_weights)
                pred_a=tf.matmul(nu_mat[n]*y_train_list[n], local_clf.predict(local_conv.predict(x_train_list[n])), transpose_a=True)/(np.sum(nu_mat[n]*y_train_list[n], 0).reshape(-1,1)+1E-15)
                pred_b=tf.matmul(nu_mat[n]*y_train_list[n], local_conv.predict(x_train_list[n]), transpose_a=True)/(np.sum(nu_mat[n]*y_train_list[n], 0).reshape(-1,1)+1E-15)
                scaled_clf_preds_list.append(cls_scaling_factor_list[n].reshape(-1,1)*pred_a)
                scaled_conv_preds_list.append(cls_scaling_factor_list[n].reshape(-1,1)*pred_b)


            print("\nServer")
            #to get the average over all the local model, we simply take the sum of the scaled weights
            average_conv_weights = sum_scaled_weights(scaled_conv_weight_list)
            average_clf_weights = sum_scaled_weights(scaled_clf_weight_list)
            average_clf_preds = np.sum(scaled_clf_preds_list, 0)
            average_conv_preds = np.sum(scaled_conv_preds_list, 0)
            #update global model
            global_conv.set_weights(average_conv_weights)
            global_clf.set_weights(average_clf_weights)

            y_pred_test=global_clf.predict(global_conv.predict(x_test))
            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            print(acc_list)
        return global_conv, global_clf
    
class FedMD:
    def __init__(self, N, l_round, comm_round, conv_choice, clf_choice, ref_choice):
        self.N=N
        self.l_round=l_round
        self.comm_round=comm_round
        self.conv_choice = conv_choice
        self.clf_choice = clf_choice
        self.ref_choice = ref_choice
    def train_model(self, D_mat, x_train_list, y_train_list, x_ref, y_ref, x_test, y_test):
        # Instantiate global_model and scaling weights for all clients
        scaling_factor_list=D_mat.sum(1)/D_mat.sum()
        pre_dist_acc_list=[]
        acc_list=[]
        global_conv=clone_model(self.conv_choice)
        global_ref=clone_model(self.ref_choice)
        global_clf=clone_model(self.clf_choice)
        # Data generator for training data
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        train_generator = ImageDataGenerator()

        # Generate training batches
        train_batches = train_generator.flow(x_ref, y_ref, batch_size=batch_size)

        for epoch in range(100):
            gc.collect()
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step in range(len(x_ref)//batch_size):
                (x_batch, y_batch) = next(train_batches)

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                tf.random.set_seed(3)
                with tf.GradientTape(persistent=True) as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    y_pred = global_ref(global_conv(x_batch, training=True), training=True)  # predictions for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch, y_pred)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads_conv = tape.gradient(loss_value, global_conv.trainable_weights)
                grads_ref = tape.gradient(loss_value, global_ref.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads_conv, global_conv.trainable_weights))
                optimizer.apply_gradients(zip(grads_ref, global_ref.trainable_weights))
                
                # Log every 10 batches.
                if step % 10 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))
 
        for t in range(self.comm_round):
            print("\nGlobal round %d" % (t,))
            optimizer.learning_rate=lr_scheduler(t)
            scaled_conv_weight_list=[]
            scaled_ref_weight_list=[]
            scaled_clf_weight_list=[]
            scaled_pred_list=[]
            
            for n in range(self.N):
                print("\nClient %d" % (n,))
                local_conv=clone_model(self.conv_choice)
                local_ref=clone_model(self.ref_choice)
                local_clf=clone_model(self.clf_choice)
                local_conv.set_weights(global_conv.get_weights())
                local_ref.set_weights(global_ref.get_weights())
                local_clf.set_weights(global_clf.get_weights())  
                # Data generator for training data
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                train_generator = ImageDataGenerator()
                if t>0:
                    # Generate reference batches
                    dist_batches = train_generator.flow(x_ref, average_preds, batch_size=batch_size)

                    #train global model
                    for epoch in range(3):
                        gc.collect()
                        print("\nStart of epoch %d" % (epoch,))

                        # Iterate over the batches of the dataset.
                        for step in range(len(x_ref)//batch_size):
                            (x_batch, y_batch) = next(dist_batches)
                            tf.random.set_seed(3)
                            with tf.GradientTape(persistent=True) as tape:
                                z_pred_ref = local_conv(x_batch, training=True)
                                y_pred_ref = local_ref(z_pred_ref, training=True)
                                # Compute the loss value for this minibatch.
                                loss_value = loss_fn(y_batch, y_pred_ref)
                            grads_conv = tape.gradient(loss_value, local_conv.trainable_weights)
                            grads_ref = tape.gradient(loss_value, local_ref.trainable_weights)
                            optimizer.apply_gradients(zip(grads_conv, local_conv.trainable_weights))
                            optimizer.apply_gradients(zip(grads_ref, local_ref.trainable_weights))

                            # Log every 10 batches.
                            if step % 10 == 0:
                                print(
                                    "Training loss (for one batch) at step %d: %.4f"
                                    % (step, float(loss_value))
                                )
                                print("Seen so far: %s samples" % ((step + 1) * batch_size))
    
                
                # Generate training batches
                train_batches = train_generator.flow(x_train_list[n], y_train_list[n], batch_size=batch_size)

                for epoch in range(self.l_round):
                    gc.collect()
                    print("\nStart of epoch %d" % (epoch,))

                    # Iterate over the batches of the dataset.
                    for step in range(len(x_train_list[n])//batch_size):
                        (x_batch, y_batch) = next(train_batches)

                        # Open a GradientTape to record the operations run
                        # during the forward pass, which enables auto-differentiation.
                        tf.random.set_seed(3)
                        with tf.GradientTape(persistent=True) as tape:

                            # Run the forward pass of the layer.
                            # The operations that the layer applies
                            # to its inputs are going to be recorded
                            # on the GradientTape.
                            z_pred = local_conv(x_batch, training=True)
                            y_pred = local_clf(z_pred, training=True)

                            # Compute the loss value for this minibatch.
                            loss_value = loss_fn(y_batch, y_pred)

                        # Use the gradient tape to automatically retrieve
                        # the gradients of the trainable variables with respect to the loss.
                        grads_conv = tape.gradient(loss_value, local_conv.trainable_weights)
                        grads_clf = tape.gradient(loss_value, local_clf.trainable_weights)
                        optimizer.apply_gradients(zip(grads_conv, local_conv.trainable_weights))
                        optimizer.apply_gradients(zip(grads_clf, local_clf.trainable_weights))

                        # Log every 10 batches.
                        if step % 10 == 0:
                            print(
                                "Training loss (for one batch) at step %d: %.4f"
                                % (step, float(loss_value))
                            )
                            print("Seen so far: %s samples" % ((step + 1) * batch_size))

                scaled_preds = custom_T(local_ref.predict(local_conv.predict(x_ref)))*scaling_factor_list[n]
                scaled_conv_weights = scale_model_weights(local_conv.get_weights(), scaling_factor_list[n])
                scaled_ref_weights = scale_model_weights(local_ref.get_weights(), scaling_factor_list[n])
                scaled_clf_weights = scale_model_weights(local_clf.get_weights(), scaling_factor_list[n])
                scaled_pred_list.append(scaled_preds)
                scaled_conv_weight_list.append(scaled_conv_weights)
                scaled_ref_weight_list.append(scaled_ref_weights)
                scaled_clf_weight_list.append(scaled_clf_weights)

            print("\nServer")
            #to get the average over all the local model, we simply take the sum of the scaled predictions
            average_preds = np.sum(scaled_pred_list, axis=0)
            average_conv_weights = sum_scaled_weights(scaled_conv_weight_list)
            average_ref_weights = sum_scaled_weights(scaled_ref_weight_list)
            average_clf_weights = sum_scaled_weights(scaled_clf_weight_list)
            global_conv.set_weights(average_conv_weights)
            global_ref.set_weights(average_ref_weights)
            global_clf.set_weights(average_clf_weights)
            
            y_pred_test=global_clf.predict(global_conv.predict(x_test))

            print('Test Classification')
            print(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            acc_list.append(bootstrap_score(y_test, np.argmax(y_pred_test,1)))
            print(acc_list)
            
        return global_conv, global_clf