import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
import pdb

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[np.int(np.floor((result.size/2))):]

def predict_1D_fbm(dx, stepsActual, reg_model):
    dx = autocorr(dx[0, :, 0])
    dx = np.reshape(dx, [1, np.size(dx), 1])
    pX = reg_model.predict(dx)
    return pX

def classification_on_real(dx, steps=50,fbm=False, model=None):
    if model is None:
        net_file = 'models/{}_new_model.h5'.format(steps)
        model = load_model(net_file)
    
    if fbm:
        fbm_model = load_model('models/{}_fbm_alpha.h5'.format(steps))
    
    dummy = np.zeros((1,steps-1,1))
    probs = []
    N = dx.shape[0]
    for j in range(max(N - steps, 1)): # average along time axis for probability estimation
        dummy[0,:,:] = np.reshape(dx[j:j+steps-1], (steps-1, 1))
        y_pred = model.predict(dummy) # get the results for 1D 

        ymean = np.mean(y_pred,axis=0) # calculate mean prediction of N-dimensional trajectory
        probs.append(ymean)
    ymean = np.mean(probs, axis = 0) # average over time
    prediction = np.argmax(ymean,axis=0) # translate to classification
    return ymean, prediction

def alpha_on_real(dx, prediction, fbm_alpha, ctrw_alpha, steps=50):
    dummy = np.zeros((1,steps-1,1))
    alphas = []
    N = dx.shape[0]
    for j in range(max(N - steps, 1)):
        dummy[0,:,:] = np.reshape(dx[j:j+steps-1], (steps-1, 1))
        if prediction == 'fbm':
            model = fbm_alpha
            y_pred = predict_1D_fbm(dummy, steps, model)
#             print(y_pred)
            value = (y_pred * 2)[0][0]
        elif prediction == 'ctrw':
            model = ctrw_alpha
            y_pred = model.predict(dummy)
            ymean = np.mean(y_pred, axis=0)
#             print(ymean)
            value = ymean[0]
        else:
            value = 1
        alphas.append(value)
#     print(alphas)
    return np.mean(alphas)

def get_activations(dx, steps=50,fbm=False, model=None):
    N=np.shape(dx)[0]
    if model is None:
        net_file = 'models/{}_new_model.h5'.format(steps)
        model = load_model(net_file)

    layer_name = 'dense_2'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    
    activations = []
    values = []
    for j in range(N):
        dummy = np.zeros((1,steps-1,1))
        dummy[0,:,:] = np.reshape(dx[j,:], (steps-1, 1))
        
        activations.append(intermediate_layer_model.predict(dummy)) # get the results for 1D
        
    return activations
def generate_dx(x):
    temp_x = x-np.mean(x)
    dx = np.diff(temp_x)
    dx = dx/np.std(dx)
    return dx