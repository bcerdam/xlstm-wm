import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, latent_dim, codes_per_latent, embedding_dim) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        self.codes_per_latent = codes_per_latent
        self.embedding_dim = embedding_dim
        self.concat_dim = latent_dim*codes_per_latent+embedding_dim

        self.linear_group_1 = nn.Sequential(nn.Linear(in_features=self.concat_dim, out_features=self.embedding_dim), 
                                           nn.LayerNorm(normalized_shape=self.embedding_dim), 
                                           nn.ReLU())

        self.linear_group_2 = nn.Sequential(nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim), 
                                            nn.LayerNorm(normalized_shape=self.embedding_dim), 
                                            nn.ReLU())

        self.linear_3 = nn.Linear(in_features=self.embedding_dim, out_features=1)

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        state_value = self.linear_group_1(state)
        state_value = self.linear_group_2(state_value)
        state_value = self.linear_3(state_value)
        return state_value