# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:03:29 2024

@author: naftabi
"""

import random
import numpy as np
from collections import deque

class Memory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state)
    
    def __len__(self):
        return len(self.buffer)