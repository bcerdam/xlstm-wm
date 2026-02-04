import torch


def total_loss(reconstruction_loss:torch.Tensor, 
               reward_loss:torch.Tensor, 
               termination_loss:torch.Tensor, 
               dynamics_loss:torch.Tensor, 
               representation_loss:torch.Tensor, 
               batch_size:int, 
               sequence_length:int) -> torch.Tensor:
    
    sum_of_losses = (reconstruction_loss+reward_loss+termination_loss+dynamics_loss+representation_loss)
    mean_loss = (1/(batch_size*sequence_length)) * sum_of_losses
    return mean_loss