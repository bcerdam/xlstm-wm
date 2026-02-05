import torch
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.sequence_preprocessing.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM


def total_loss_step(reconstruction_loss:torch.Tensor, 
                    reward_loss:torch.Tensor, 
                    termination_loss:torch.Tensor, 
                    dynamics_loss:torch.Tensor, 
                    representation_loss:torch.Tensor, 
                    batch_size:int, 
                    sequence_length:int, 
                    dynamics_beta:float, 
                    representations_beta:float, 
                    categorical_encoder:CategoricalEncoder, 
                    categorical_decoder:CategoricalDecoder, 
                    tokenizer:Tokenizer, 
                    dynamics_model:XLSTM_DM, 
                    optimizer:torch.optim.Optimizer, 
                    scaler:torch.amp.grad_scaler) -> torch.Tensor:
            
    sum_of_losses = (reconstruction_loss+reward_loss+termination_loss+dynamics_beta*dynamics_loss+representations_beta*representation_loss)
    mean_loss = (1/(batch_size*sequence_length)) * sum_of_losses

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(mean_loss).backward()
    scaler.unscale_(optimizer)
    
    torch.nn.utils.clip_grad_norm_(categorical_encoder.parameters(), 1000.0)
    torch.nn.utils.clip_grad_norm_(categorical_decoder.parameters(), 1000.0)
    torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1000.0)
    torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), 1000.0)
    
    scaler.step(optimizer)
    scaler.update()

    return mean_loss