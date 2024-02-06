import torch
import numpy as np

class MoconTask:
    def __init__(
        self,
        num_parallel: int,
        device: str,
        npz_path: str,
        mvae_path: str
    ):
        self.num_parallel = num_parallel
        self.device = device

        self.load_mocap_and_mvae(npz_path, mvae_path)

        self.action_scale = 4.0
        self.dt = 0.01
        self.frame_size = self.mvae_motion.shape[1]
        self.action_size = self.mvae_model.latent_size
        self.max_timestep = 1800

        self.num_condition_frames = self.mvae_model.num_condition_frames
        self.history_size = 5
        assert (
            self.history_size >= self.num_condition_frames
        ), "History size must be not less than condition size."
        self.history = torch.zeros(
            (self.num_parallel, self.history_size, self.frame_size)
        ).to(self.device)

        self.start_indices = torch.randint(
            low=0, high=self.mvae_motion.shape[0], size=(self.num_parallel,)
        ).long().to(self.device)
        self.root_facing = torch.zeros(
            (self.num_parallel, 1,)
        ).to(self.device)
        self.root_xz = torch.zeros(
            (self.num_parallel, 2,)
        ).to(self.device)
        self.vel_xz = torch.zeros(
            (self.num_parallel, 2,)
        ).to(self.device)

        # control param.
        target_size = 2
        self.target = torch.zeros(
            (self.num_parallel, target_size,)
        ).to(self.device)
        self.target_direct = torch.zeros(
            (self.num_parallel, 1,)
        ).to(self.device)
        self.ctrl_condition_size = (
            self.frame_size * self.num_condition_frames
        ) + target_size

    @torch.no_grad()
    def load_mocap_and_mvae(self, npz_path, mvae_path):
        mvae_mocap = np.load(npz_path)
        self.num_not_ee = mvae_mocap["num_not_ee"]

        self.mvae_motion = mvae_mocap["mvae_motion"]
        self.mvae_motion = torch.from_numpy(self.mvae_motion).float().to(self.device)

        self.mvae_model = torch.load(mvae_path, map_location=self.device)
        self.mvae_model.eval()

    def get_2d_rotmat(self, yaw) -> torch.Tensor:
        yaw = -yaw # reverse for xz coord. angle express
        col1 = torch.cat((yaw.cos(), yaw.sin()), dim=-1)
        col2 = torch.cat((-yaw.sin(), yaw.cos()), dim=-1)
        rotmat = torch.stack((col1, col2), dim=-1)
        return rotmat

    def integrate_root_info(self, next_frame):
        next_frame = next_frame[:, 0, :]

        rotmat = self.get_2d_rotmat(self.root_facing)
        local_vel_xz = next_frame[..., :2]
        vel_xz = (
            rotmat * local_vel_xz.unsqueeze(dim=-1)
        ).sum(dim=2)
        self.root_xz.add_(vel_xz * self.dt)

        avel_y = next_frame[:, 2]
        self.root_facing.add_(
            (avel_y * self.dt).unsqueeze(dim=-1)
        ).remainder_(2*np.pi)

        self.history = self.history.roll(1, dims=1)
        self.history[:, 0].copy_(next_frame)

        # foot contact
        pass

    def get_mvae_condition(self, normalize=False, flatten=True):
        condition = self.history[:, :self.num_condition_frames]
        if normalize:
            condition = self.mvae_model.normalize(condition)
        if flatten:
            condition = condition.flatten(start_dim=1, end_dim=2)
        return condition
    
    def get_mvae_next_frame(self, action):
        # action: z
        self.action = action
        condition = self.get_mvae_condition(normalize=True)

        # with torch.no_grad():
        mvae_output = self.mvae_model.sample(
            action, condition, deterministic=True
        )
        mvae_output = mvae_output.view(
            -1,
            self.mvae_model.num_future_predictions,
            self.frame_size
        )

        next_frame = self.mvae_model.denormalize(mvae_output)
        return next_frame

    def reset_initial_frames(self):
        self.start_indices.random_(
            0, self.mvae_motion.shape[0] - self.num_condition_frames + 1
        )

        condition_range = (
            self.start_indices.repeat((self.num_condition_frames, 1)).t() +
            torch.arange(
                start=self.num_condition_frames-1,
                end=-1,
                step=-1
            ).long().to(self.device)
        )

        self.history[:, :self.num_condition_frames].copy_(
            self.mvae_motion[condition_range]
        )

    def get_target_delta_and_angle(self):
        target_delta = self.target - self.root_xz
        target_angle = self.root_facing + torch.atan2(
            target_delta[:, 1], target_delta[:, 0]
        ).unsqueeze(dim=1)
        return target_delta, target_angle
    
    def reset(self):
        self.root_xz.zero_()
        self.root_facing.zero_()
        self.timestep = 0

        self.reset_target()
        self.reset_initial_frames()

    def reset_target(self):
        facing_switch_every = 240
        
        if self.timestep % facing_switch_every == 0:
            self.target_direct.uniform_(0, 2*np.pi)

        self.target.copy_(self.root_xz)
        self.target[:, 0].add_(10 * self.target_direct.cos().squeeze())
        self.target[:, 1].add_(10 * self.target_direct.sin().squeeze())

    def get_controller_input(self):
        condition = self.get_mvae_condition(normalize=False)
        _, target_angle = self.get_target_delta_and_angle()
        ctrl_input = torch.cat(
            (condition, target_angle.cos(), target_angle.sin()),
            dim=-1
        )
        return ctrl_input

    def get_next_frame(self, action):
        # action = action * self.action_scale
        next_frame = self.get_mvae_next_frame(action)
        return next_frame

    def update(self, action):
        next_frame = self.get_next_frame(action)
        self.integrate_root_info(next_frame)
        self.timestep += 1
