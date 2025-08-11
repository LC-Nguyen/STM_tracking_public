from utils import generate
import numpy as np
batchsize=32
steps=40
T= np.arange(19,21,0.1)
sigma=0
dilation=3
gen = generate(batchsize=batchsize,steps=steps,T=T,sigma=sigma,dilation=dilation,interpolate=1)
next(gen)
