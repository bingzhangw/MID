import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb

class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'

            )
        )

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z
    
    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        #print(f"Using {sampling}")
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        
        # NodeType: PEDESTRIAN
        # predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)  

        # NodeType: VEHICLE
        # Assume velocity (v_x, v_y) has shape [batch_size, horizon, 2]
        v_x, v_y = predicted_y_vel[..., 0], predicted_y_vel[..., 1]

        speed = torch.sqrt(v_x**2 + v_y**2)          # Linear speed
        angle = torch.atan2(v_y, v_x)                # Heading angle

        dt = 0.5  

        dphi = (angle[..., 1:] - angle[..., :-1]) / dt
        a = (speed[..., 1:] - speed[..., :-1]) / dt

        # Pad the first timestep to keep shapes consistent
        dphi = torch.cat([dphi[..., :1], dphi], dim=-1)
        a = torch.cat([a[..., :1], a], dim=-1)

        controls = torch.stack([dphi, a], dim=-1)  # [batch_size, num_samples, horizon, 2]

        predicted_y_pos = dynamics.integrate_samples(controls)  


        return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        feat_x_encoded = self.encode(batch,node_type) # B * 64
        loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        return loss
