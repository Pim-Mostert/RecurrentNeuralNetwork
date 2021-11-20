import numpy as np
import matplotlib.pyplot as plt
from empty import Empty
from sigmoid import *
from generate_data import *

numT = 50000
data = generate_data(4)

alpha = 0.001
numK = 3

w = np.random.randn(numK)
b = np.random.randn(numK)
v = np.random.randn(numK, numK)
u = np.random.randn(numK)
d = np.random.randn(1)

log = Empty()
log.y_hat = np.zeros([numT])
log.x = np.zeros([numT])
log.y = np.zeros([numT])
log.w = np.zeros([numT, numK]); 
log.b = np.zeros([numT, numK]); 
log.v = np.zeros([numT, numK, numK]); 
log.u = np.zeros([numT, numK]); 
log.d = np.zeros([numT]); 
log.mse = np.zeros([numT])

h = np.zeros([numK])
y = next(data)
mse = 1

for it in range(0, numT):
    h_tmin1 = h
    
    # Forward
    x = y
    a = w*x + (v*h[:, None]).sum(axis=0) + b
    h = sigmoid(a)
    y_hat = (u * h).sum() + d
      
    # Backward
    y = next(data)
    delta = -(y - y_hat)
    
    dl_dd = delta
    dl_du = delta * h
    dl_db = delta * u * dsigmoid(a)
    dl_dw = delta * u * dsigmoid(a) * x
    dl_dv = delta * (h_tmin1[:, None] * ((u * dsigmoid(a))[None, :]))
    
    # Learn
    w -= alpha*dl_dw
    v -= alpha*dl_dv
    b -= alpha*dl_db
    u -= alpha*dl_du
    d -= alpha*dl_dd

    mse = (1-alpha)*mse + alpha*(delta**2)

    # Log
    log.w[it, :] = w
    log.v[it, :, :] = v
    log.u[it, :] = u
    log.b[it, :] = b
    log.d[it] = d
    log.x[it] = x
    log.y_hat[it] = y_hat
    log.y[it] = y
    log.mse[it] = mse
    
    if ((it % 1000) == 0):
        print(f'Finished {it}/{numT}')
    
plt.figure()
plt.plot(np.log(log.mse))

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(log.y)
plt.plot(log.y_hat)
plt.legend(['y','y_hat'])
plt.xlim([0, 100])
plt.subplot(2, 1, 2)
plt.plot(log.y)
plt.plot(log.y_hat)
plt.legend(['y','y_hat'])
plt.xlim([numT-100, numT])

# plt.figure()
# plt.plot(log.w)
# plt.plot(log.u)
# plt.plot(log.v.reshape([numT, -1]))
# plt.plot(log.b)
# plt.plot(log.d)
# plt.legend(['w', 'u', 'v', 'b', 'd'])    

    
    
    
    