import torch.nn as nn


class ActorCriticNetwork(nn.Module):

    def __init__(self, engine_input_dim, msn_input_dim, factory_input_dim,
                 state_input_dim, hidden_dim=64):
        super().__init__()
        self.engine = nn.Sequential(
            nn.Linear(engine_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.msn = nn.Sequential(
            nn.Linear(msn_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.factory = nn.Sequential(
            nn.Linear(factory_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_engines(self, feats):
        return self.engine(feats).squeeze(-1)

    def score_msns(self, feats):
        return self.msn(feats).squeeze(-1)

    def score_factories(self, feats):
        return self.factory(feats).squeeze(-1)

    def state_value(self, state):
        return self.value(state).squeeze(-1)
