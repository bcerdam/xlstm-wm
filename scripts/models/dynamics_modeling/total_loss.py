import torch
import itertools
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM


def total_loss_step(reconstruction_loss:torch.Tensor, 
                    reward_loss:torch.Tensor, 
                    termination_loss:torch.Tensor, 
                    dynamics_loss:torch.Tensor,
                    categorical_encoder:CategoricalEncoder, 
                    categorical_decoder:CategoricalDecoder, 
                    tokenizer:Tokenizer, 
                    dynamics_model:XLSTM_DM, 
                    optimizer:torch.optim.Optimizer, 
                    scaler:torch.amp.grad_scaler) -> torch.Tensor:
            
    # sum_of_losses = (reconstruction_loss+reward_loss+termination_loss+dynamics_loss)

    # optimizer.zero_grad(set_to_none=True)
    # scaler.scale(sum_of_losses).backward()
    # scaler.unscale_(optimizer)
    
    # all_parameters = itertools.chain(
    #     categorical_encoder.parameters(),
    #     categorical_decoder.parameters(),
    #     tokenizer.parameters(),
    #     dynamics_model.parameters()
    # )
    # torch.nn.utils.clip_grad_norm_(all_parameters, 1000.0, foreach=True)
    
    # scaler.step(optimizer)
    # scaler.update()

    # return sum_of_losses
    sum_of_losses = reconstruction_loss + reward_loss + termination_loss + dynamics_loss

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(sum_of_losses).backward()
    scaler.unscale_(optimizer)
    
    params = [p for group in optimizer.param_groups for p in group['params']]
    torch.nn.utils.clip_grad_norm_(params, 1000.0, foreach=True)
    
    scaler.step(optimizer)
    scaler.update()

    return sum_of_losses
    
