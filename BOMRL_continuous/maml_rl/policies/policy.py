import torch
import torch.nn as nn

from maml_rl.utils.torch_utils import weighted_mean, to_numpy, detach_distribution
from torch.distributions.kl import kl_divergence

from collections import OrderedDict

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # For compatibility with Torchmeta
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def update_params(self, reinforce_loss, episodes, policy, params=None, step_size=0.1, first_order=True, algorihtm_index=1):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        #if params is None:
        #    params = OrderedDict(self.named_meta_parameters())

        params_meta = OrderedDict(self.named_meta_parameters())

        pi1 = policy(episodes.observations, params=params)
        old_pi1 = detach_distribution(pi1)

        optimizer = torch.optim.Adam(params.values(), lr=0.001) # 0.00001 for 2D navigation, 0.001 for locomotion tasks

        for i in range(50):
            optimizer.zero_grad()
            inner_loss = reinforce_loss(policy,episodes,params=params)
            if algorihtm_index==1:
                kls = weighted_mean(kl_divergence(policy(episodes.observations, params=params), old_pi1),lengths=episodes.lengths).mean()
            elif algorihtm_index==2:
                kls = weighted_mean(kl_divergence(old_pi1,policy(episodes.observations, params=params)),lengths=episodes.lengths).mean()
            else:
                assert algorihtm_index == 1   

            loss = inner_loss / (step_size * 5.0) + kls # 5.0 for locomotion tasks, 0.8 for 2D navigation task
            #print("total_loss ", loss)
            #print("get_kl ",kls)
            if kls.clone().detach().numpy()>10.0:
                print("please increase the regularzation weight \lambda or reduce the lower-level optimization learning rate")
                #break
                
            loss.backward()
            optimizer.step()

        #print("get_kl ",kls)
            
        updated_params = OrderedDict()
        for (name, param), (name_meta, param_meta) in zip(params.items(), params_meta.items()):
            updated_params[name] = param_meta - param_meta.data + param.data
        return updated_params
