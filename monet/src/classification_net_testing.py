"""
classification_net_testing.py
This script contains functions for activating and testing of the classification
net.
"""  
 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
from utils import fbm_diffusion,Brownian,CTRW,OrnsteinUng,generate_sim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scipy.io
import pickle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style="white", palette="muted", color_codes=True)


"""
Function classification_on_file is used to classify trajectories loaded from a
.mat file

The function assumes the input comes in the form x,y,z,...,N where N is the 
trajectory serial number, starting from one.

Input: 
    file - string containing the file name, ending with .mat
    
Outputs:
    prediction - Classification to diffusion model type where 0=FBM; 1=Brownian; 2=CTRW
    y_full - matrix of network probabilities. Each trajectory recieves 3 values 
             which are the probabilities of being assigned to a specific
             diffusion model.
"""


def classification_on_sim():
    dx,label=generate_sim(batchsize=100,steps=50,T=15,sigma=0.01)
    with open("example_track.txt", "w") as f:
        # np.set_printoptions(threshold=sys.maxsize)
        f.write(np.array2string(dx, threshold=np.inf))
    ### change here to load a different network model
    print(label)
    N=np.shape(dx)[0]
    net_file = './Models/50_model.h5'
    model = load_model(net_file)
    predictions = []
    for j in range(N):
        dummy = np.zeros((1,49,1))
        dummy[0,:,:] = dx[j,:,:]
        y_pred = model.predict(dummy) # get the results for 1D 
        ymean = np.mean(y_pred,axis=0) # calculate mean prediction of N-dimensional trajectory 
        prediction = np.argmax(ymean,axis=0) # translate to classification
        predictions.append(prediction)
        print('y_pred {}'.format(y_pred))
        print('prediction {}'.format(prediction))
        print('ground truth',label[j])
        print('--')

    accuracy = 100 * np.sum([1 if label[i] == predictions[i] else 0 for i in range(len(label))]) / len(label)
    print(accuracy)
    return

def classification_on_real():
    dx = np.loadtxt("em18tracks.txt")
    # dx,label=generate_sim(batchsize=100,steps=50,T=15,sigma=0.01)
    ### change here to load a different network model
    N=np.shape(dx)[0]
    net_file = './Models/30_new_model.h5'
    model = load_model(net_file)
    predictions = []
    for j in range(N):
        dummy = np.zeros((1,29,1))
        dummy[0,:,:] = np.reshape(dx[j,:], (29, 1))
        y_pred = model.predict(dummy) # get the results for 1D 
        ymean = np.mean(y_pred,axis=0) # calculate mean prediction of N-dimensional trajectory 
        prediction = np.argmax(ymean,axis=0) # translate to classification
        predictions.append(prediction)
        print('y_pred {}'.format(y_pred))
        print('prediction {}'.format(prediction))
    # print('ground truth',label[j])
    # print('--')

    # accuracy = 100 * np.sum([1 if label[i] == predictions[i] else 0 for i in range(len(label))]) / len(label)
    # print(accuracy)
    return
    
def classification_on_file(file):
    ### change here to load a different network model
    net_file = './Models/model_testsigma0-wB-6layern-300-variance-FBMCTRW.h5'
    model = load_model(net_file)

    # load mat file and extract trajectory data
    
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
    
    numTraj = len(np.unique(data[:,NAxes]))
    prediction = np.zeros([numTraj,1])
    y_full = np.zeros([numTraj,3])
    flag = np.zeros([numTraj,1])
    for i in np.arange(1,numTraj+1):
        y_pred = np.zeros([NAxes,3])
        for j in range(NAxes):
            x = data[np.argwhere(data[:,NAxes]==i),j]
            x = x-np.mean(x)
            if len(x)>=300: # classification network is trained on 300 step trajectories
                flag[i-1] = 1 # mark trajectories that are being analyzed
                dx = np.diff(x,axis=0)
                variance= np.std(dx)
                dx =dx/variance
                dx = np.reshape(dx[:299],[1,299,1]) #change this number based on step size
            y_pred[j,:] = model.predict(dx) # get the results for 1D 
        ymean = np.mean(y_pred,axis=0) # calculate mean prediction of N-dimensional trajectory 
        prediction[i-1,0] = np.argmax(ymean,axis=0) # translate to classification
        y_full[i-1,:] = ymean
    prediction = prediction[np.where(flag==1)]
    
    return prediction,y_full



classification_on_real()
# prediction,y_full = classification_on_file(file = './data/trajInterleaved_matrix.mat')
# print(y_full)
# print(prediction.shape)
# print(prediction)
# pickle.dump( prediction, open( "./results/pred-syntraj.p", "wb" ) )
# pickle.dump( y_full, open( "./results/yfull-syntraj.p", "wb" ) )