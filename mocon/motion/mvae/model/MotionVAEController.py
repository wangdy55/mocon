import torch.nn as nn

from mocon.motion.motion_vae.train.train_util import init

class MotionVAEController(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        init_r_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        init_s_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("sigmoid"),
        )
        init_t_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("tanh"),
        )

        h_size = 256
        self.actor = nn.Sequential(
            init_r_(nn.Linear(self.input_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_t_(nn.Linear(h_size, self.output_size)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.actor(x)