# replay buffer for SMB game

import numpy as np 
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        '''
        Add a new experience to the replay buffer
        '''
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        
    #
    def sample(self, batch_size):
        '''
        Sample a batch of experiences from the replay buffer
        '''
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
        
    # get the length of the buffer
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        
        