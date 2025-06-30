# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:11:49 2024

@author: naftabi
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim

import random

from .utils import Memory
from .model import QNetwork

class DQLAgent:
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 64, 
                 lr: float = 1e-2, 
                 gamma: float = 0.99,
                 tau: float = 0.05,
                 update_every: int = 10,
                 epsilon_start: float = 0.95,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 seed: Optional[int] = None):
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        self.gamma = gamma
        self.tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.update_every = update_every
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=lr)
        self.memory = Memory(buffer_size)
        self.criterion = nn.HuberLoss()
        self.losses = []
        self.q_values = []
        
    def update_epsilon(self):
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)  
        
    def act(self, state):
        self.q_network.eval()
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_value = self.q_network.forward(state)
            action = q_value.max(1)[1].item()
            self.q_values.append(q_value.max().item())
        else:
            action = random.randrange(self.action_dim)
        self.update_epsilon()
        return action
    
    def update(self, batch_size):
        self.q_network.train()
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        
        q_values = self.q_network(state)
        with torch.no_grad():
            next_q_values = self.target_q_network(next_state)
        
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        max_next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * max_next_q_value * (1 - done)
        
        loss = self.criterion(q_value, expected_q_value)
        self.losses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            for target_params, params in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                target_params.data.copy_(self.tau * params.data + (1.0 - self.tau) * target_params.data)
                
    def save_policy(self, idx):
        torch.save(
            self.q_network.state_dict(),
            './saved_q/agent_{}.pkl'.format(idx)
            )
    
    def load_policy(self, idx):
        self.q_network.load_state_dict(
            torch.load('./saved_q/agent_{}.pkl'.format(idx))
            )