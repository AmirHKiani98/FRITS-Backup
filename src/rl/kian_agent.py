# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:11:49 2024

@author: naftabi
"""

from typing import Optional
from src.models.fedlight.model import ActorNetwork, CriticNetwork
import torch.optim as optim
from torch import nn, FloatTensor, no_grad, distributions, stack, tensor, log
import torch
class KianAgent:
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 64, 
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.001,
                 seed: Optional[int] = None
                 ):
        
        self.state_dim = state_dim
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        
    def act(self, state):
        self.actor.eval()
        state = FloatTensor(state).unsqueeze(0)
        with no_grad():
            probs = self.actor(state)
        dist = distributions.Categorical(probs)
        action = dist.sample().item()
        return action
    
    def update(self, trajectory):
        states, actions, td_targets, advantages = zip(*trajectory)

        states = stack(states)
        actions = tensor(actions)
        td_targets = tensor(td_targets)
        advantages = tensor(advantages)

        # Update actor
        log_probs = log(self.actor(states).gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(log_probs * advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        values = self.critic(states).squeeze()
        critic_loss = ((td_targets - values) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def get_gradients(self):
        actor_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.actor.parameters()]
        critic_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.critic.parameters()]
        return {
            'actor': actor_grads,
            'critic': critic_grads
        }
    
    def step(self):
        """
        Apply previously computed gradients via optimizer step.
        Call this after applying federated averaged gradients.
        """
        self.actor_optimizer.step()
        self.critic_optimizer.step()
    
    def compute_gradients(self, trajectory):
        """
        Compute actor and critic gradients from a given trajectory.
        This method zeroes out gradients and performs .backward(),
        but does not apply optimizer.step().
        Call `step()` separately after applying federated averaged gradients.
        """
        import torch
        from torch import stack, tensor

        if not trajectory:
            return

        # Unpack trajectory
        states, actions, td_targets, advantages = zip(*trajectory)
        states = stack(states)  # shape: [batch_size, state_dim]
        actions = tensor(actions)
        td_targets = tensor(td_targets)
        advantages = tensor(advantages)

        # Actor loss
        action_probs = self.actor(states)  # shape: [batch_size, num_actions]
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(action_log_probs * advantages).mean()

        # Critic loss (mean squared TD error)
        values = self.critic(states).squeeze()  # V(s)
        critic_loss = ((td_targets - values) ** 2).mean()

        # Backpropagation
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # retain if you want to inspect later

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Return loss values for optional logging
        return actor_loss.item(), critic_loss.item()


    def apply_gradients(self, avg_grads):
        for p, g in zip(self.actor.parameters(), avg_grads['actor']):
            p.grad = g
        self.actor_optimizer.step()

        for p, g in zip(self.critic.parameters(), avg_grads['critic']):
            p.grad = g
        self.critic_optimizer.step()
                
    def save_policy(self, idx):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, f'./saved_q/agent_fedlight_{idx}.pth')

    def load_policy(self, idx):
        checkpoint = torch.load(f'./saved_q/agent_fedlight_{idx}.pth')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])