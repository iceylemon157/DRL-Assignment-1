# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

pickle_file = "q_table.pkl"
try:
    with open(pickle_file, "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    print(f"File {pickle_file} not found. Using random actions.")
    q_table = {}

def get_state_key(obs):
    """
    Create a unique key for the state based on its attributes.
    This is useful for storing Q-values in a dictionary.
    """
    return obs

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    state = get_state_key(obs)

    if state not in q_table:
        return random.choice([0, 1, 2, 3, 4, 5])

    return np.argmax(q_table[state])

    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

