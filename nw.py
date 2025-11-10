import torch.distributions
import torch.nn as nn

# fully-connected categorical actor
class FCCA(nn.Module):
    def __init__(self,
                 state_dim,
                 n_actions,
                 hidden_dims=(128,64)):
        super().__init__()
        layers_sizes = [state_dim, *hidden_dims]
        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], n_actions))
        self.body = nn.Sequential(*layers)

    def forward(self, state):
        return self.body(state)

    def full_pass(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy

    def greedy_action(self, state):
        logits = self.forward(state)
        return logits.argmax(dim=-1)

class FCV(nn.Module):
    def __init__(self,
                 state_dim,
                 hidden_dims=(128, 64)):
        super().__init__()
        layers_sizes = [state_dim, *hidden_dims]
        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.body = nn.Sequential(*layers)

    def forward(self, state):
        return self.body(state).squeeze(-1)

