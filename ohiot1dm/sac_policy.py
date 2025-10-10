import d3rlpy
import numpy as np


trained_model_pat1 = d3rlpy.load_learnable("./Models/sac_model_pat1.d3")

def sac_policy_pat1(state):
    
    state = np.array(state).reshape(1, -1)  
    action = trained_model_pat1.predict(state)  
    return action[0] 


trained_model_pat2 = d3rlpy.load_learnable("./Models/sac_model_pat2.d3")

def sac_policy_pat2(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat2.predict(state)
    return action[0]


trained_model_pat3 = d3rlpy.load_learnable("./Models/sac_model_pat3.d3")

def sac_policy_pat3(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat3.predict(state)
    return action[0]


trained_model_pat4 = d3rlpy.load_learnable("./Models/sac_model_pat4.d3")

def sac_policy_pat4(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat4.predict(state)
    return action[0]


trained_model_pat5 = d3rlpy.load_learnable("./Models/sac_model_pat5.d3")

def sac_policy_pat5(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat5.predict(state)
    return action[0]


trained_model_pat6 = d3rlpy.load_learnable("./Models/sac_model_pat6.d3")

def sac_policy_pat6(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat6.predict(state)
    return action[0]


trained_model_pat7 = d3rlpy.load_learnable("./Models/sac_model_pat7.d3")

def sac_policy_pat7(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat7.predict(state)
    return action[0]


trained_model_pat8 = d3rlpy.load_learnable("./Models/sac_model_pat8.d3")

def sac_policy_pat8(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat8.predict(state)
    return action[0]


trained_model_pat9 = d3rlpy.load_learnable("./Models/sac_model_pat9.d3")

def sac_policy_pat9(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat9.predict(state)
    return action[0]


trained_model_pat10 = d3rlpy.load_learnable("./Models/sac_model_pat10.d3")

def sac_policy_pat10(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat10.predict(state)
    return action[0]


trained_model_pat11 = d3rlpy.load_learnable("./Models/sac_model_pat11.d3")

def sac_policy_pat11(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat11.predict(state)
    return action[0]


trained_model_pat12 = d3rlpy.load_learnable("./Models/sac_model_pat12.d3")

def sac_policy_pat12(state):
    state = np.array(state).reshape(1, -1)
    action = trained_model_pat12.predict(state)
    return action[0]