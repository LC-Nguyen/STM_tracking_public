import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,BatchNormalization,Conv1D
from tensorflow.keras.layers import Input,GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from tensorflow.keras.optimizers import Adam
from utils import generate
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,CSVLogger
import datetime
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import pickle

def create_ctrw_model(steps,net_file,temp_range=[19,21],epochs=50):
    history = History()
    
    batchsize = 32
    T = np.arange(temp_range[0],temp_range[1],0.1) # this provides another layer of stochasticity to make the network more robust
    steps = steps
     # number of steps to generate
    initializer = 'he_normal'
    f = 32 #number of filters
    sigma = 0 #noise variance
    
    inputs = Input((steps-1,1))
    
    x1 = Conv1D(f,2,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f,2,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f,2,dilation_rate=4,padding='same',activation='relu',kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling1D()(x1)
    
    
    x2 = Conv1D(f,2,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f,2,dilation_rate=3,padding='same',activation='relu',kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f,2,dilation_rate=5,padding='same',activation='relu',kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling1D()(x2)
    
    
    x3 = Conv1D(f,3,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f,3,dilation_rate=1,padding='same',activation='relu',kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = GlobalAveragePooling1D()(x3)
    
    
    x4 = Conv1D(f,3,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f,3,dilation_rate=1,padding='same',activation='relu',kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f,3,dilation_rate=3,padding='same',activation='relu',kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = GlobalAveragePooling1D()(x4)
    
    
    x5 = Conv1D(f,4,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x5 = BatchNormalization()(x5)
    x5 = Conv1D(f,4,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x5)
    x5 = BatchNormalization()(x5)
    x5 = Conv1D(f,4,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x5)
    x5 = BatchNormalization()(x5)
    x5 = GlobalAveragePooling1D()(x5)
    
    x7 = Conv1D(f,4,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x7 = BatchNormalization()(x7)
    x7 = Conv1D(f,4,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x7)
    x7 = BatchNormalization()(x7)
    x7 = Conv1D(f,4,dilation_rate=3,padding='same',activation='relu',kernel_initializer=initializer)(x7)
    x7 = BatchNormalization()(x7)
    x7 = GlobalAveragePooling1D()(x7)
    
    
    x6 = Conv1D(f,3,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x6 = BatchNormalization()(x6)
    x6 = GlobalAveragePooling1D()(x6)
    
    
    con = concatenate([x1,x2,x3,x4,x5,x6])
    dense = Dense(512,activation='relu')(con)
    dense = Dense(256,activation='relu')(dense)
    dense = Dense(128,activation='relu')(dense)
    dense = Dense(64,activation='relu')(dense)
    dense = Dense(32,activation='relu')(dense)
    dense2 = Dense(3,activation='softmax')(dense)
    model = Model(inputs=inputs, outputs=dense2)
    
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])  # Changed 'acc' to 'accuracy'
    model.summary()
    
    
    callbacks = [
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               min_lr=1e-9),
             ModelCheckpoint(filepath=net_file,
                             monitor='val_accuracy',  # Changed 'val_acc' to 'val_accuracy'
                             save_best_only=False,
                             mode='max',
                             save_weights_only=False), history]
    
    
    gen = generate(batchsize=batchsize,steps=steps,T=T,sigma=sigma)
    model.fit(x=gen,
            steps_per_epoch=50,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=generate(batchsize=batchsize,steps=steps,T=T,sigma=sigma),
            validation_steps=25)
        
    
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_acc  = history.history['accuracy']      # Changed 'acc' to 'accuracy'
    val_acc    = history.history['val_accuracy']  # Changed 'val_acc' to 'val_accuracy'
    xc         = range(epochs)
    
    
    ##https://stackoverflow.com/questions/11026959/writing-a-dict-to-txt-file-and-reading-it-back
    with open(f'{steps}_{temp_range[0]}_{temp_range[1]}_ep{epochs}_history.pkl', 'wb') as handle:
        pickle.dump(history.history, handle)
    
    plt.figure()
    plt.plot(xc, train_loss, label="train loss")
    plt.plot(xc, val_loss, label="validation loss")
    plt.legend()
    plt.savefig(f"{steps}_{temp_range[0]}_{temp_range[1]}_ep{epochs}_loss.png")