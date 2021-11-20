import numpy as np
import matplotlib.pyplot as plt
from Common import *

def run(cfg0, x, y):    
    w = np.random.randn(1)
    b = np.random.randn(1)
    v = cfg0.v0
    u = np.random.randn(1)
    d = np.random.randn(1)
    
    log = Empty()
    log.y_hat = np.zeros([cfg0.numN, cfg0.numT])
    log.w = np.zeros([cfg0.numN]); 
    log.b = np.zeros([cfg0.numN]); 
    log.v = np.zeros([cfg0.numN]); 
    log.u = np.zeros([cfg0.numN]); 
    log.d = np.zeros([cfg0.numN]); 
    log.mse = np.zeros([cfg0.numN])
    
    for n in range(0, cfg0.numN):
        # Forward
        a = np.zeros([cfg0.numT])
        h = np.zeros([cfg0.numT])
        y_hat = np.zeros([cfg0.numT])
        
        a[0] = w*x[0] + b
        h[0] = sigmoid(a[0])
        y_hat[0] = u*h[0] + d
            
        for it in range(1, cfg0.numT):
            a[it] = w*x[it] + v*h[it-1] + b
            h[it] = sigmoid(a[it])
            y_hat[it] = u*h[it] + d
          
        # Backward
        delta = -(y - y_hat)
        
        dh_dv = np.zeros([cfg0.numT])
        dh_dw = np.zeros([cfg0.numT])
        dh_db = np.zeros([cfg0.numT])
        
        dh_dv[0] = 0
        dh_dw[0] = dsigmoid(a[0])*x[0]
        dh_db[0] = dsigmoid(a[0])
        for it in range(1, cfg0.numT):
            dh_dv[it] = dsigmoid(a[it])*(h[it-1] + v*dh_dv[it-1])
            dh_dw[it] = dsigmoid(a[it])*(x[it] + v*dh_dw[it-1])
            dh_db[it] = dsigmoid(a[it])*(1 + v*dh_db[it-1])
            
        dv = (delta*u*dh_dv).sum() / cfg0.numT
        dw = (delta*u*dh_dw).sum() / cfg0.numT
        db = (delta*u*dh_db).sum() / cfg0.numT
        
        du = (delta*h).sum() / cfg0.numT
        dd = (delta).sum() / cfg0.numT
        
        # Learn
        w -= cfg0.alpha*dw
        v -= cfg0.alpha_v*dv
        u -= cfg0.alpha*du
        b -= cfg0.alpha*db
        d -= cfg0.alpha*dd
    
        # Log
        log.w[n] = w
        log.v[n] = v
        log.u[n] = u
        log.b[n] = b
        log.d[n] = d
        log.y_hat[n] = y_hat
        log.mse[n] = (delta**2).sum() / cfg0.numN
        
        if ((cfg0.feedback_epoch is not None) and (n % cfg0.feedback_epoch) == 0):
            print(f'Finished {n}/{cfg0.numN}')
            
    return log
    

    
    
    
    
    
    
    
    
    
    
    