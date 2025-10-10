import d3rlpy
import numpy as np


trained_model = d3rlpy.load_learnable("sac_model_toy.d3")

def sac_policy(state):
    
    state = np.array(state).reshape(1, -1)  
    action = trained_model.predict(state)  
    return action[0] 