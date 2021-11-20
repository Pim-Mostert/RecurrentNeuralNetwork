import numpy as np

def generate_data(period):
    x = 0
    
    while True:
        yield np.sin(x * (2*np.pi)/period) \
            + np.sin(x * (2*np.pi)/(period*5)) \
            + np.sin(x * (2*np.pi)/(period*20))
        x += 1