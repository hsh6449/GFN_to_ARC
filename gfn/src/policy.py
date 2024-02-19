import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax

# import segmentation_models_pytorch as smp
import numpy as np
import pdb


class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim*state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_actions)
        
        # self.conv2d = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=1)
        # self.relu = nn.ReLU()
        # self.Unet = smp.Unet(
        #     encoder_name="resnet18",
        #     encoder_weights=None,
        #     in_channels=1,
        #     classes=1,
        # )

        # self.decode = nn.Conv2d(3,1, kernel_size=3, stride=1)

        self.coordinate = [0,-1]

    def forward(self, s):
        # try :
        x = torch.flatten(s, 0)
        # 이부분 바꾸는게 좋다고 함 clone().detach()로? 근데 dtype때문에 이렇게 한거라서 일단 놔둠
        x = x.to(torch.float32)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)


        predicted_mask = self.select_mask(s, "one")


        return softmax(x, dim=0), predicted_mask
        
        
    def select_mask(self, s, mode="Unet"):
        
        if mode == "whole":
            ## 전체 grid 마스크 선택
            selection = np.zeros((30, 30), dtype=bool)
            selection[:s.shape[0], :s.shape[1]] = np.ones(s.shape, dtype=bool)

        elif mode == "one":
            # selection = np.zeros((30, 30), dtype=bool)

            self.coordinate[1] += 1

            if self.coordinate[1] == 5: # size 바꾸면 여기 바뀌여야함 나중에 객체 지향으로 수정할 것
                self.coordinate[1] = 0
                self.coordinate[0] += 1

            coordinate = self.coordinate

            if self.coordinate[0] == 4 & self.coordinate[1] == 4 : 
                self.coordinate = [0,-1]

                

            # selection[self.coordinate[0], self.coordinate[1]] = 1

            return coordinate


            
        """elif mode == "Unet":
                
            if type(s) is not torch.float :
                s = s.to(torch.float)

            s = self.conv2d(s.unsqueeze(0))
            s = self.relu(s)

            out = self.Unet(s.reshape(3,1,32,32))

            out = self.decode(out.reshape(3,32,32))
            out = self.relu(out)

            out = out.sigmoid()
            pred_mask = (out >= 0.6).float()
            pred_mask = pred_mask.squeeze()

            return np.array(pred_mask.detach().cpu(), dtype=bool)"""

class BackwardPolicy(nn.Module):
    def __init__(self, state_dim,hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim*state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_actions)
        

    def forward(self, s):
        x = torch.flatten(s, 0)
            # 이부분 바꾸는게 좋다고 함 clone().detach()로? 근데 dtype때문에 이렇게 한거라서 일단 놔둠
        x = x.to(torch.float32)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)


        return softmax(x, dim=0)

