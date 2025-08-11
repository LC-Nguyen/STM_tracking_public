from scipy.io import loadmat
import numpy as np
def mat_read(file):
    ### change here to load a different network model
    # net_file = './Models/model_testsigma0-wB-6layern-300-variance-FBMCTRW.h5'
    # model = load_model(net_file)

    # load mat file and extract trajectory data
    
    f = loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        print(varName)
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
    print(data)
    numTraj = len(np.unique(data[:,NAxes]))
    prediction = np.zeros([numTraj,1])
    y_full = np.zeros([numTraj,3])
    flag = np.zeros([numTraj,1])
mat_read('./data/trajInterleaved_matrix.mat')