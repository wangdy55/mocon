import torch
import torch.nn as nn
import torch.nn.functional as F

class ConMVAE(nn.Module):
    def __init__(
            self,
            signal_size,
            mvae,
        ):
        super().__init__()
        self.mvae = mvae
        frame_size = mvae.frame_size
        latent_size = mvae.latent_size
        hidden_size = 256
        self.fc1 = nn.Linear(frame_size+signal_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, latent_size)

    def act(self, u, c):
        h1 = F.relu(self.fc1(torch.cat((u, c), dim=1)))
        h2 = F.relu(self.fc2(h1))
        z = self.fc4(h2)
        return z
    
    def decode(self, z, c):
        return self.mvae.sample(z, c, deterministic=True)

    def forward(self, u, c):
        z = self.act(u, c)
        # TODO: Process action z 
        return self.decode(z, c)