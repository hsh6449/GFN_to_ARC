import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as pp
from .log_s import Log



class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env=None, device='cuda'):
        super().__init__()
        self.total_flow = Parameter(torch.ones(1))

        self.forward_policy = forward_policy.to(device)
        self.backward_policy = backward_policy.to(device)
        self.env = env
        self.device = device
        self.actions = {}


        self.patches = {
  'smile': lambda: pp.gca().add_patch(pp.Polygon(np.stack([np.linspace(0.2,0.8), 0.3-np.sin(np.linspace(0,3.14))*0.15]).T, closed=False, fill=False, lw=3)),
  'frown': lambda: pp.gca().add_patch(pp.Polygon(np.stack([np.linspace(0.2,0.8), 0.15+np.sin(np.linspace(0,3.14))*0.15]).T, closed=False, fill=False, lw=3)),
  'left_eb_down': lambda: pp.gca().add_line(pp.Line2D([0.15, 0.35], [0.75,0.7], color=(0,0,0))),
  'right_eb_down': lambda: pp.gca().add_line(pp.Line2D([0.65, 0.85], [0.7,0.75], color=(0,0,0))),
  'left_eb_up': lambda: pp.gca().add_line(pp.Line2D([0.15, 0.35], [0.7,0.75], color=(0,0,0))),
  'right_eb_up': lambda: pp.gca().add_line(pp.Line2D([0.65, 0.85], [0.75,0.7], color=(0,0,0))),
}
        self.sorted_keys = sorted(self.patches.keys())

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
        

    def forward_probs(self, s):
        """
        Returns a vector of probabilities over actions in a given state.

        Args:
            s: An NxD matrix representing N states
        """
        probs = self.forward_policy(s)

        return probs


    def sample_states(self, s0, return_log=True):
        """
        Samples and returns a collection of final states from the GFlowNet.

        Args:
            s0: An NxD matrix of initial states

            return_log: Return an object containing information about the
            sampling process (e.g. the trajectory of each sample, the forward
            and backward probabilities, the actions taken, etc.)
        """
        

        log = Log(s0, self.backward_policy, self.total_flow, self.patches, self.sorted_keys) if return_log else None
        is_done = False

        iter = 0
    

        while not is_done:
            iter += 1
            
            probs_s = self.forward_probs(self.face_to_tensor(s0))        
            prob = Categorical(logits = probs_s)
            ac = prob.sample()


            new_state = s0 + [self.sorted_keys[ac]]

            parent_states, parent_actions = self.face_parents(new_state)

            px = torch.stack([self.face_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()

            # parent_edge_flow_preds = self.forward_probs(px)[torch.arange(len(parent_states)), pa]


            if iter == 3 :
                reward = self.face_reward(new_state)
                # and since there are no children to this state F(s,a) = 0 \forall a
                edge_flow_prediction = torch.zeros(6)
            else:
            # Otherwise we keep going, and compute F(s, a)
                reward = 0
                edge_flow_prediction = self.forward_probs(self.face_to_tensor(new_state))

            s0 = new_state

            if return_log:
                log.log(s=s0, probs=prob.log_prob(ac).squeeze(), actions = ac, rewards=reward, total_flow=self.total_flow, done=is_done)  # log에 저장

            if iter >= 3:  # max_length miniarc = 25, arc = 900
                return (s0, log) if return_log else s0
            
            # if selection[0] == 4 & selection[1] == 4:
            #     return (s, log) if return_log else s

            if is_done:
                break

        # return (s, log) if return_log else s
        return new_state, log if return_log else new_state

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
    
    def face_reward(self,face):
        if self.has_overlap(face):
            return 0
        eyebrows = 'left_eb_down', 'left_eb_up', 'right_eb_down', 'right_eb_up'
        # Must have exactly two eyebrows
        if sum([i in face for i in eyebrows]) != 2:
            return 0
        # We want twice as many happy faces as sad faces so here we give a reward of 2 for smiles
        if 'smile' in face:
            return 2
        if 'frown' in face:
            return 1  # and a reward of 1 for frowns
        # If we reach this point, there's no mouth
        return 0
    
    def has_overlap(self,face):
  # Can't have two overlapping eyebrows!
        if 'left_eb_down' in face and 'left_eb_up' in face:
            return True
        if 'right_eb_down' in face and 'right_eb_up' in face:
            return True
        # Can't have two overlapping mouths!
        if 'smile' in face and 'frown' in face:
            return True
        return False
    
    def face_parents(self,state):
        parent_states = []  # states that are parents of state
        parent_actions = []  # actions that lead from those parents to state
        for face_part in state:
            # For each face part, there is a parent without that part
            parent_states.append([i for i in state if i != face_part])
            # The action to get there is the corresponding index of that face part
            parent_actions.append(self.sorted_keys.index(face_part))
        return parent_states, parent_actions
    

    def face_to_tensor(self, face):
        return torch.tensor([i in face for i in self.sorted_keys]).float()


