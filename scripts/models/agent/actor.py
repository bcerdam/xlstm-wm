import torch
import torch.nn as nn


class EMAScalar:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = None

    def __call__(self, x):
        self.value = x if self.value is None else self.decay * self.value + (1 - self.decay) * x
        return self.value

def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage * len(flat_x))
    return torch.kthvalue(flat_x, kth).values


def actor_loss(batch_lambda_returns:torch.Tensor, 
               state_values:torch.Tensor, 
               log_policy:torch.Tensor, 
               nabla:float, 
               entropy:torch.Tensor, 
               lowerbound_ema:EMAScalar,
               upperbound_ema:EMAScalar) -> float:
    
    # advantage = (batch_lambda_returns - state_values).detach()
    lower_bound = lowerbound_ema(percentile(batch_lambda_returns, 0.05))
    upper_bound = upperbound_ema(percentile(batch_lambda_returns, 0.95))
    scale = torch.max(torch.ones(1, device=batch_lambda_returns.device), upper_bound - lower_bound)
    advantage = (batch_lambda_returns - state_values).detach() / scale
    
    loss = -1*advantage*log_policy - nabla*entropy
    return loss.mean()


class Actor(nn.Module):
    def __init__(self, latent_dim, codes_per_latent, embedding_dim, env_actions) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.codes_per_latent = codes_per_latent
        self.embedding_dim = embedding_dim
        self.concat_dim = latent_dim*codes_per_latent+embedding_dim
        self.env_actions = env_actions

        self.linear_group_1 = nn.Sequential(nn.Linear(in_features=self.concat_dim, out_features=self.embedding_dim), 
                                            nn.LayerNorm(normalized_shape=self.embedding_dim), 
                                            nn.ReLU())

        self.linear_group_2 = nn.Sequential(nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim), 
                                            nn.LayerNorm(normalized_shape=self.embedding_dim), 
                                            nn.ReLU())

        self.linear_3 = nn.Linear(in_features=self.embedding_dim, out_features=self.env_actions)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        logits = self.linear_group_1(state)
        logits = self.linear_group_2(logits)
        logits = self.linear_3(logits)
        return logits
