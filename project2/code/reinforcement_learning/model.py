import torch
import torch.nn as nn
import torch.nn.functional as F
from obs import State 

class DQNet(nn.Module):
    def __init__(self, map_in_size, global_in_size):
        '''init the Model
        Args:
            map_in_size (int): map input size
            global_in_size (int): global info size
        '''
        super(DQNet, self).__init__()

        self.l_dep=9
        
        self.feature_size =32
        self.global_feature_size = 10

        self.conv_input = nn.Conv2d(in_channels=map_in_size, out_channels=self.feature_size, kernel_size=(3, 3),padding=(1,1))
        
        self.globalL = nn.Linear(global_in_size, self.global_feature_size, bias=True)
        
        self.combinL = nn.Linear(self.feature_size + self.global_feature_size, self.feature_size, bias=True) 
        
        self.convblocks = nn.ModuleList([nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=(3, 3),padding=(1,1)) for _ in range(self.l_dep)])
        
        self.actL = nn.Linear(self.feature_size, 3, bias=False)
    
    def forward(self, state: State):
                
        # Get rotated four input
        map_input = state[0].float()
        rotated_input = [map_input]
        for _ in range(3):
            rotated_input.append(torch.rot90(rotated_input[-1], 1, [2, 3]))
        
        # global information input
        global_input = state[1].float()
        g = F.relu_(self.globalL(global_input))
        repeated_g = g.repeat_interleave(self.feature_size * self.feature_size, 0)
        repeated_g = repeated_g.view(g.size(0), self.feature_size, self.feature_size, g.size(1))

        rotated_f = []
        for i in range(4):
            h = F.relu_(self.conv_input(rotated_input[i]))
            for block in self.convblocks:
                h = F.relu_(h + block(h))
            h = h.permute([0, 2, 3, 1])
            h = torch.cat((h, repeated_g), 3)
            h = F.relu_(self.combinL(h))
            h = self.actL(h) # shape: (batch_size, 32, 32, 3)
            rotated_f.append(h)
        
        # Get the output of actions
        dir_move=[]
        St_move=[]
        Bd_move=[]
        for i in range(4):
            dir_move.append(rotated_f[i][..., :1].rot90(-i, [1, 2]))
            St_move.append(rotated_f[i][..., 1:2].rot90(-i, [1, 2]))
            Bd_move.append(rotated_f[i][..., 2:3].rot90(-i, [1, 2]))
        St_move = torch.cat(St_move, dim=3)
        St_move = torch.mean(St_move, dim=3, keepdim=True)
        Bd_move = torch.cat(Bd_move, dim=3)
        Bd_move = torch.mean(Bd_move, dim=3, keepdim=True)
        out_actions_map = torch.cat((dir_move[0], dir_move[1], dir_move[2], dir_move[3], St_move, Bd_move), dim=3)
        return out_actions_map
