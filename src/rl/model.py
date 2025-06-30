# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:06:06 2024

@author: naftabi
"""

import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        
    def forward(self, state):
        x = F.selu(self.fc1(state))
        x = self.norm1(x)
        x = F.selu(self.fc2(x))
        x = self.norm2(x)
        return self.fc3(x)  # No activation function for Q-values
