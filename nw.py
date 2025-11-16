import torch.distributions as D
import torch.nn as nn
import torch

from constants import TOTAL_LAYERS, TOTAL_MOVES



# Convolution Neural Network
class CNNActorCritic(nn.Module):
    def __init__(self,
                 in_channels=TOTAL_LAYERS,
                 n_actions=TOTAL_MOVES,
                 conv_channels=(256, 256),
                 shared_hidden=(512,),
                 actor_hidden=(256, 256),
                 critic_hidden=(256, 256),
                 convolution_activation_function=nn.ReLU,
                 fully_connected_activation_function=nn.ReLU,
                 actor_activation_function=nn.ReLU,
                 critic_activation_function=nn.ReLU):
        super().__init__()

        # Общее тело со свёрткой
        conv_layers = []
        c_in = in_channels
        for c_out in conv_channels:
            conv_layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            conv_layers.append(convolution_activation_function())
            c_in = c_out
        self.conv = nn.Sequential(*conv_layers)

        # Общие полносвязные слои
        self.flatten = nn.Flatten()
        shared_layers = []
        in_dim = c_in * 8 * 8
        for h in shared_hidden:
            shared_layers.append(nn.Linear(in_dim, h))
            shared_layers.append(fully_connected_activation_function())
            in_dim = h
        self.shared = nn.Sequential(*shared_layers)
        self.shared_out_dim = in_dim

        # actor head
        actor_layers = []
        in_dim = self.shared_out_dim
        for h in actor_hidden:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(actor_activation_function())
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, n_actions))
        self.actor = nn.Sequential(*actor_layers)

        # critic head
        critic_layers = []
        in_dim = self.shared_out_dim
        for h in critic_hidden:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(critic_activation_function())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        critic_layers.append(nn.Tanh())
        self.critic = nn.Sequential(*critic_layers)

    def _features(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.shared(x)
        return x

    def forward(self, x):
        feat = self._features(x)
        logits = self.actor(feat)
        values = self.critic(feat).squeeze(-1)
        return logits, values

    def greedy_action(self, x):
        feat = self._features(x)
        logits = self.actor(feat)
        return logits.argmax(dim=-1)

    def values_only(self, x):
        feat = self._features(x)
        values = self.critic(feat).squeeze(-1)
        return values

    # Тут не используется маска легальных ходов!!!
    def logits_only(self, x):
        feat = self._features(x)
        logits = self.actor(feat)
        return logits

    # Это жадный ход, который используется,
    # когда нейросеть является оппонентом
    @torch.no_grad()
    def make_move(self, x, legal_mask):
        device = next(self.parameters()).device

        single_obs = False

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)

        if x.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            single_obs = True
            x = x.unsqueeze(0)

        logits = self.logits_only(x)

        if not isinstance(legal_mask, torch.Tensor):
            legal_mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
        else:
            legal_mask = legal_mask.to(device)

        if legal_mask.dim() == 1:
            # (A) -> (1, A)
            legal_mask = legal_mask.unsqueeze(0)

        logits[~legal_mask] = -1e9
        actions = logits.argmax(dim=-1)

        if single_obs:
            return int(actions.item())

        return actions