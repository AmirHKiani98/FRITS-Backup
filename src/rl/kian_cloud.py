import torch


class KianLightCloud:
    def __init__(self):
        self.actor_buffer = []
        self.critic_buffer = []

    def collect(self, agent_id, gradients):
        self.actor_buffer.append(gradients['actor'])
        self.critic_buffer.append(gradients['critic'])

    def average_and_dispatch(self):
        avg_actor = self.average_gradients(self.actor_buffer)
        avg_critic = self.average_gradients(self.critic_buffer)
        self.actor_buffer.clear()
        self.critic_buffer.clear()
        return {'actor': avg_actor, 'critic': avg_critic}
    
    def average_gradients(self, gradient_list):
        """
        Averages a list of gradients from multiple agents.
        Each gradient list element is a list of tensors (per-layer gradients).
        
        Args:
            gradient_list: List of list of tensors (one list per agent)

        Returns:
            List of averaged tensors
        """
        num_agents = len(gradient_list)
        if num_agents == 0:
            raise ValueError("No gradients provided for averaging.")

        avg_grads = []
        for grads_per_layer in zip(*gradient_list):
            stacked = torch.stack(grads_per_layer)  # shape: [num_agents, *param_shape]
            mean_grad = torch.mean(stacked, dim=0)
            avg_grads.append(mean_grad)

        return avg_grads