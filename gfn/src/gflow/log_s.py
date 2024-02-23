import torch
import torch.nn as nn

from torch.distributions import Categorical

class Log:
    def __init__(self, s0, backward_policy, total_flow, patches, sorted_keys):
        """
        Initializes a Stats object to record sampling statistics from a
        GFlowNet (e.g. trajectories, forward and backward probabilities,
        actions, etc.)

        Args:
            s0: The initial state of collection of samples

            backward_policy: The backward policy used to estimate the backward
            probabilities associated with each sample's trajectory

            total_flow: The estimated total flow used by the GFlowNet during
            sampling

            env: The environment (i.e. state space and reward function) from
            which samples are drawn
        """
        self.patch = patches
        self.sorted_keys = sorted_keys
        self.backward_policy = backward_policy
        self.total_flow = total_flow
        self._traj = []  # state value
        self._traj.append(s0)
        self._fwd_probs = []
        self._back_probs = []
        self._actions = []
        self._state_colors = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []

    def log(self, s, probs, actions, rewards=None, total_flow=None, done=None):
        """
        Logs relevant information about each sampling step

        Args:
            s: An NxD matrix containing he current state of complete and
            incomplete samples

            probs: An NxA matrix containing the forward probabilities output by the
            GFlowNet for the given states

            actions: A Nx1 vector containing the actions taken by the GFlowNet
            in the given states

            done: An Nx1 Boolean vector indicating which samples are complete
            (True) and which are incomplete (False)
        """
        
        self._traj.append(s)
        self._fwd_probs.append(probs.unsqueeze(0))
        self._actions.append(actions)
        self.rewards.append(rewards)
        self.total_flow = total_flow

    def face_to_tensor(self, face):
            return torch.tensor([i in face for i in self.sorted_keys]).float().to("cuda")


    @property
    def traj(self):
        if type(self._traj) is list:
            # self._traj = torch.cat(self._traj, dim=1)[:, :-1, :]
            pass

        # terminal = torch.tensor(self.env.unwrapped.answer)
        # pad_terminal = torch.zeros(30,30)
        # pad_terminal[:terminal.shape[0], :terminal.shape[1]] = terminal
        
        # self._traj[-1] = torch.tensor(pad_terminal, dtype=int)
        return self._traj

    @property
    def fwd_probs(self):
        if type(self._fwd_probs) is list:
            # self._fwd_probs = torch.cat(self._fwd_probs, dim=0)  # 여기 뭐가 들어가야하지
            pass
        # half_length = len(self._fwd_probs) // 2
        # import pdb; pdb.set_trace()
        return self._fwd_probs

    @property
    def actions(self):
        if type(self._actions) is list:
            # self._actions = torch.cat(self._actions, dim=1)
            pass
        return self._actions

    @property
    def back_probs(self):
        # if self._back_probs is not None:
        #     return self._back_probs

        # s = self.traj[-2]  # 마지막에서 두번째 꺼
        # prev_s = self.traj[-1]  # 마지막 꺼
        # if type(self._back_probs) is list:
        #     pass
        # actions = self.actions[-1]
        # import pdb; pdb.set_trace()
        # terminated = (actions == 34) | (len(self.actions) == 100)

        
        
        for t in range(3):
            # if t == 0:
            #     self._back_probs.append(torch.tensor(1.0))
            Pb_s = self.backward_policy(self.face_to_tensor(self.traj[-t]))
            pb = Categorical(logits = Pb_s).log_prob(self.actions[-t])
            self._back_probs.append(pb.unsqueeze(0))        
            
        # import pdb;pdb.set_trace()
        return self._back_probs
