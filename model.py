import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, state_shape, n_actions, backbone_hidden=[64, 64], V=False, Q=False):
        super().__init__()

        self.state_shape = state_shape[0]
        self.n_actions = n_actions
        self.V = V
        self.Q = Q

        self.backbone = nn.Sequential()
        for i, features in enumerate(zip([self.state_shape] + backbone_hidden[:-1], backbone_hidden)):
            in_features, out_features = features
            self.backbone.add_module('layer_' + str(i), nn.Linear(in_features, out_features))
            self.backbone.add_module('relu_' + str(i), nn.ReLU())

        self.policy = nn.Sequential()
        self.policy.add_module('policy_layer_1', nn.Linear(backbone_hidden[-1], 8 * self.state_shape))
        self.policy.add_module('policy_relu', nn.ReLU())
        self.policy.add_module('policy_layer_2', nn.Linear(8 * self.state_shape, self.n_actions))

        if self.V:
            self.v = nn.Sequential()
            self.v.add_module('v_layer_1', nn.Linear(backbone_hidden[-1], 8 * self.state_shape))
            self.v.add_module('v_relu', nn.ReLU())
            self.v.add_module('v_layer_2', nn.Linear(8 * self.state_shape, 1))

        if self.Q:
            self.q = nn.Sequential()
            self.q.add_module('q_layer_1', nn.Linear(backbone_hidden[-1], 8 * self.state_shape))
            self.q.add_module('q_relu', nn.ReLU())
            self.q.add_module('q_layer_2', nn.Linear(8 * self.state_shape, self.n_actions))

    def forward(self, states):
        result = {}

        x = self.backbone(states)
        
        result['logits'] = self.policy(x)
        if self.V:
            result['V'] = self.v(x)
        if self.Q:
            result['Q'] = self.q(x)
        

