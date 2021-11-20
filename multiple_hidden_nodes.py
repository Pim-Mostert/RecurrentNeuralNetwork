import numpy as np
import matplotlib.pyplot as plt
from empty import Empty
from sigmoid import *

numT = 99

data = (1/3)*np.sin(np.arange(0, numT+1)*(2*np.pi*10)/(numT+1)) \
    + (1/3)*np.sin(np.arange(0, numT+1)*(2*np.pi*2)/(numT+1)) \
    + (1/3)*(np.arange(0, numT+1) * (3/(numT+1)))
x = data[0:-1]
y = data[1:]

alpha = 0.5
numN = 50000
numK = 5

w = np.random.randn(numK)
b = np.random.randn(numK)
v = np.random.randn(numK, numK)
u = np.random.randn(numK)
d = np.random.randn(1)

log = Empty()
log.y_hat = np.zeros([numN, numT])
log.w = np.zeros([numN, numK]); 
log.b = np.zeros([numN, numK]); 
log.v = np.zeros([numN, numK, numK]); 
log.u = np.zeros([numN, numK]); 
log.d = np.zeros([numN]); 
log.mse = np.zeros([numN])

for n in range(0, numN):
    # Forward
    a = np.zeros([numT, numK])
    h = np.zeros([numT, numK])
    y_hat = np.zeros([numT])
    
    a[0, :] = w*x[0] + b
    h[0, :] = sigmoid(a[0, :])
    y_hat[0] = (u * h[0, :]).sum() + d
        
    for it in range(1, numT):
        a[it, :] = w*x[it] \
            + (v*h[it-1][:, None]).sum(axis=0) \
            + b
        h[it, :] = sigmoid(a[it, :])
        y_hat[it] = (u * h[it, :]).sum() + d
      
    # Backward
    delta = -(y - y_hat)
    
    dh_db = np.zeros([numT, numK, numK])        # numT * b_k * h_j
    dh_dv = np.zeros([numT, numK, numK, numK])  # numT * v_k'k * h_j
    dh_dw = np.zeros([numT, numK, numK])        # numT * w_k* h_j
        
    dh_db[0, :, :] = dsigmoid(a[0, :])[None, :]
    dh_dw[0, :, :] = (dsigmoid(a[0, :])*x[0])[None, :]
    dh_dv[0, :, :, :] = 0
    for it in range(1, numT):
        dh_db[it] = ((v[None, :, :] * dh_db[it-1, :, :][:, :, None]).sum(axis=1) \
            + 1) * dsigmoid(a[it])
        dh_dw[it] = ((v[None, :, :] * dh_dw[it-1, :, :][:, :, None]).sum(axis=1) \
            + x[it]) * dsigmoid(a[it])
        dh_dv[it] = ((dh_dv[it-1, :, :, :][:, :, :, None] * v[None, None, :, :]).sum(axis=2) \
            + h[it-1, :][:, None, None]) * dsigmoid(a[it])[None, None, :]
                     
    db_t = (u[None, None, :] * dh_db).sum(axis=2) * delta[:, None]
    dw_t = (u[None, :] * dh_dw).sum(axis=2) * delta[:, None]
    dv_t = (u[None, None, None, :] * dh_dv).sum(axis=3) * delta[:, None, None]
    
    du_t = delta[:, None] * h
    dd_t = delta
    
    # Learn
    w -= alpha*dw_t.sum(axis=0) / numT
    v -= alpha*dv_t.sum(axis=0) / numT
    b -= alpha*db_t.sum(axis=0) / numT
    u -= alpha*du_t.sum(axis=0) / numT
    d -= alpha*dd_t.sum(axis=0) / numT

    # Log
    log.w[n, :] = w
    log.v[n, :, :] = v
    log.u[n, :] = u
    log.b[n, :] = b
    log.d[n] = d
    log.y_hat[n] = y_hat
    log.mse[n] = (delta**2).sum() / numN
    
    if ((n % 100) == 0):
        print(f'Finished {n}/{numN}')
    
plt.plot(np.log(log.mse))

plt.figure()
plt.plot(y)
plt.plot(log.y_hat[numN-1, :])
plt.legend(['y','y_hat'])

plt.figure()
plt.plot(log.w)
plt.plot(log.u)
plt.plot(log.v.reshape([numN, -1]))
plt.plot(log.b)
plt.plot(log.d)
plt.legend(['w', 'u', 'v', 'b', 'd'])    

plt.figure()
plt.scatter(x, y_hat)
    
    
    
    
    
    
    
    
    
    
    
    