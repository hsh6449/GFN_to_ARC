import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical, Normal

import numpy as np
from .log import Log

import segmentation_models_pytorch as smp
from collections import OrderedDict

class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env=None, device='cuda'):
        super().__init__()
        self.total_flow = Parameter(torch.tensor(1.0).to(device))
        # self.std = Parameter(torch.tensor([0.5]*10, dtype = torch.float32).to(device))
        self.forward_policy = forward_policy.to(device)
        self.backward_policy = backward_policy.to(device)
        self.env = env
        self.device = device
        self.actions = {}

        self.env_style = "bbox" # "point", ...

        """
        Initializes a GFlowNet using the specified forward and backward policies
        acting over an environment, i.e. a state space and a reward function.

        Args:
            forward_policy: A policy network taking as input a state and
            outputting a vector of probabilities over actions

            backward_policy: A policy network (or fixed function) taking as
            input a state and outputting a vector of probabilities over the
            actions which led to that state

            env: An environment defining a state space and an associated reward
            function"""
        
        self.conv2d = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=1).to(device)
        self.relu = nn.ReLU()
        self.Unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=1,
            classes=1,
        ).to(device)

        self.decode = nn.Conv2d(3,1, kernel_size=3, stride=1).to(device)

    def forward_probs(self, s):
        """
        Returns a vector of probabilities over actions in a given state.

        Args:
            s: An NxD matrix representing N states
        """
        probs = self.forward_policy(s)

        return probs
    
    def unflatten_vec(obs,grid_size):
        s = grid_size*grid_size
        sizes = [s, 2, s, 2, s, 2,   1, s, s, 2, 2, s, 1,     s, 1, 1]
        sizesum = [0]
        for _s in sizes:
            sizesum.append(sizesum[-1]+_s)
        
        names = [
            "clip","clip_dim",
            "grid","grid_dim",
            "input","input_dim",
                "active",
                "background",
                "object","object_dim","object_pos","object_sel",
                "rotation_parity",
            "selected",
            "terminated",
            "trials_remain",
            ]
        req_reshape = ["clip", "grid", "input", "background", "object", "object_sel", "selected"]

        splits = [obs[:, st:ed].long() for st, ed in zip(sizesum[:-1],sizesum[1:])]
        res =  OrderedDict(zip(names,splits))
        for k in req_reshape:
            res[k] = res[k].reshape(-1,grid_size,grid_size)
        return res

    def sample_states(self, s0,info, return_log=True):
        """
        Samples and returns a collection of final states from the GFlowNet.

        Args:
            s0: An NxD matrix of initial states

            return_log: Return an object containing information about the
            sampling process (e.g. the trajectory of each sample, the forward
            and backward probabilities, the actions taken, etc.)
        """
        # s = torch.tensor(s0["input"], dtype=torch.float).to(self.device)
        s = torch.tensor(s0, dtype=torch.float).to(self.device)
        # s = torch.tensor(s0, dtype=torch.float).to(self.device)
        grid_dim = info["input_dim"]

        log = Log(s, self.backward_policy, self.total_flow,
                  self.env) if return_log else None
        is_done = False

        iter = 0
        # selection_mode = "Unet" # one , Unet, whole

        while not is_done:
            iter += 1
            # probs = self.forward_probs(s)  # 랜덤액션?
            probs, selection = self.forward_probs(s)
            ac = int(Categorical(probs).sample())

            if self.env_style == "point":
                # ac = int(Normal(probs, self.std).rsample().argmax())
                self.actions["operation"] = ac
                self.actions["selection"] = selection  # selection 어떻게
                result = self.env.step(self.actions)
            
            elif self.env_style == "bbox" :
                # bbox 
                x0 = selection[0] * grid_dim[0]
                y0 = selection[1] * grid_dim[1]
                x1 = selection[2] * grid_dim[0]
                y1 = selection[3] * grid_dim[1]

                selection = (x0,y0,x1,y1, ac)

            
                self.actions["operation"] = selection
                result = self.env.step(self.actions["operation"])

            # reward 는 spase reward 이기 때문에 따로 reward 함수를 만들어서 log에 저장하는 함수를 만들어야함
            state, reward, is_done, _, info = result
            # s = torch.tensor(state["grid"], dtype = torch.long).to(self.device)
            s = torch.tensor(state, dtype = torch.float).to(self.device)


            re = self.reward(self.unflatten_vec(state,info["input_dim"][0]))

            ime_reward = re + reward

            if return_log:
                log.log(s=state, probs=probs, actions = self.actions, rewards=ime_reward, total_flow=self.total_flow, done=is_done)  # log에 저장

            if iter >= 100:  # max_length miniarc = 25, arc = 900
                return (s, log) if return_log else s
            
            # if selection[0] == 4 & selection[1] == 4:
            #     return (s, log) if return_log else s

            if is_done:
                break

        # return (s, log) if return_log else s
        return s, log if return_log else s

    def evaluate_trajectories(self, traj, actions):
        """
        Returns the GFlowNet's estimated forward probabilities, backward
        probabilities, and rewards for a collection of trajectories. This is
        useful in an offline learning context where samples drawn according to
        another policy (e.g. a random one) are used to train the model.

        Args:
            traj: The trajectory of each sample

            actions: The actions that produced the trajectories in traj
        """
        num_samples = len(traj)
        traj = traj.reshape(-1, traj.shape[-1])
        actions = actions.flatten()
        finals = traj[actions == len(actions) - 1]
        zero_to_n = torch.arange(len(actions))

        fwd_probs = self.forward_probs(traj)
        fwd_probs = torch.where(
            actions == -1, 1, fwd_probs[zero_to_n, actions])
        fwd_probs = fwd_probs.reshape(num_samples, -1)

        actions = actions.reshape(num_samples, -1)[:, :-1].flatten()

        back_probs = self.backward_policy(traj)
        back_probs = back_probs.reshape(num_samples, -1, back_probs.shape[1])
        back_probs = back_probs[:, 1:, :].reshape(-1, back_probs.shape[2])
        back_probs = torch.where((actions == -1) | (actions == 2), 1,
                                 back_probs[zero_to_n[:-num_samples], actions])
        back_probs = back_probs.reshape(num_samples, -1)

        rewards = self.reward(finals)

        return fwd_probs, back_probs, rewards
    
    def reward(self, s):
        """
        Returns the reward associated with a given state.

        Args:
            s: An NxD matrix representing N states
        """
        terminal = torch.tensor(self.env.unwrapped.answer).to("cuda")
        pad_terminal = torch.zeros_like(s)
        pad_terminal[:terminal.shape[0], :terminal.shape[1]] = terminal

        # MSE 
        r = ((pad_terminal - s)**2 + 1e-6) # pad_terminal은 ARC용
        for i in range(len(r)):
            for j in range(len(r[0])):
                if r[i][j] == float("inf"):
                    r[i][j] = 0
        mse_reward = 1 / (r.sum() + 1e-6) 

        # sparse reward
        # sparse_reward = torch.where((pad_terminal - s).sum(dim=1) == 0, 1, 0.1)
        return mse_reward

    def select_mask(self, s):

        if type(s) is not torch.float :
            s = s.to(torch.float)

        s = self.conv2d(s.unsqueeze(0))
        s = self.relu(s)

        out = self.Unet(s.reshape(3,1,32,32))

        out = self.decode(out.reshape(3,32,32))
        out = self.relu(out)

        out = out.sigmoid()
        pred_mask = (out >= 0.5).float()
        pred_mask = pred_mask.squeeze()

        return np.array(pred_mask.detach().cpu(), dtype=bool)
    
    ## 작은 그리드부터 해보기 
    
