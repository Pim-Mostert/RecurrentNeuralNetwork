import numpy as np
import matplotlib.pyplot as plt
from empty import Empty
from sigmoid import *

numT = 99

data = np.sin(np.arange(0, numT+1)*(2*np.pi*10)/(numT+1))
x = data[0:-1]
y = data[1:]

alpha = 1
numN = 2500

w = np.random.randn(1)
b = np.random.randn(1)
v = np.random.randn(1)
u = np.random.randn(1)
d = np.random.randn(1)

log = Empty()
log.y_hat = np.zeros([numN, numT])
log.w = np.zeros([numN]); 
log.b = np.zeros([numN]); 
log.v = np.zeros([numN]); 
log.u = np.zeros([numN]); 
log.d = np.zeros([numN]); 
log.mse = np.zeros([numN])

for n in range(0, numN):
    # Forward
    a = np.zeros([numT])
    h = np.zeros([numT])
    y_hat = np.zeros([numT])
    
    a[0] = w*x[0] + b
    h[0] = sigmoid(a[0])
    y_hat[0] = u*h[0] + d
        
    for it in range(1, numT):
        a[it] = w*x[it] + v*h[it-1] + b
        h[it] = sigmoid(a[it])
        y_hat[it] = u*h[it] + d
      
    # Backward
    delta = -(y - y_hat)
    
    dh_dv = np.zeros([numT])
    dh_dw = np.zeros([numT])
    dh_db = np.zeros([numT])
    
    dh_dv[0] = 0
    dh_dw[0] = dsigmoid(a[0])*x[0]
    dh_db[0] = dsigmoid(a[0])
    for it in range(1, numT):
        dh_dv[it] = dsigmoid(a[it])*(h[it-1] + v*dh_dv[it-1])
        dh_dw[it] = dsigmoid(a[it])*(x[it] + v*dh_dw[it-1])
        dh_db[it] = dsigmoid(a[it])*(1 + v*dh_db[it-1])
        
    dv = (delta*u*dh_dv).sum() / numT
    dw = (delta*u*dh_dw).sum() / numT
    db = (delta*u*dh_db).sum() / numT
    
    du = (delta*h).sum() / numT
    dd = (delta).sum() / numT
    
    # Learn
    w -= alpha*dw
    v -= alpha*dv
    u -= alpha*du
    b -= alpha*db
    d -= alpha*dd

    # Log
    log.w[n] = w
    log.v[n] = v
    log.u[n] = u
    log.b[n] = b
    log.d[n] = d
    log.y_hat[n] = y_hat
    log.mse[n] = (delta**2).sum() / numN
    
    if ((n % 100) == 0):
        print(f'Finished {n}/{numN}')
    
plt.plot(log.mse)

plt.figure()
plt.plot(y)
plt.plot(log.y_hat[numN-1])
plt.legend(['y','y_hat'])

plt.figure()
plt.plot(log.w)
plt.plot(log.u)
plt.plot(log.v)
plt.plot(log.b)
plt.plot(log.d)
plt.legend(['w', 'u', 'v', 'b', 'd'])    

plt.figure()
plt.scatter(x, y_hat)
    
    
    
    
    
    
    
    
    
    
    
    