import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os, time

from mocon.motion.mvae.model.ConMVAE import ConMVAE
from mocon.utils.SpringUtil import SpringUtil
from mocon.utils.QuatUtil import QuatUtil

# dir.
BVH_DIR = "mocon/motion/mocap/bvh"
NPZ_DIR = "mocon/motion/mocap/npz"
LOG_DIR = "mocon/motion/mvae/log"
MODEL_DIR = "mocon/motion/mvae/model"

def get_2d_rotmat(yaw) -> torch.Tensor:
    yaw = -yaw # reverse for xz coord. angle express
    col1 = torch.cat((yaw.cos(), yaw.sin()), dim=-1)
    col2 = torch.cat((-yaw.sin(), yaw.cos()), dim=-1)
    rotmat = torch.stack((col1, col2), dim=-1)
    return rotmat

def feed_con_mvae(con_mvae, con_signal, condition):
    next_frame = con_mvae(con_signal, condition)
    loss = (next_frame[:, 2]- con_signal[:, 2]).pow(2).mean()
    
    return next_frame, loss

def train_ctrl():
    args = SimpleNamespace(
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
        signal_size=3,
        num_frames=10e6,
        num_parallel=100,
        num_steps_per_rollout=1200,
        facing_switch_every=240
    )

    # path of log and checkpoint
    t_str = time.strftime("%y%m%d_%H%M%S", time.localtime(round(time.time())))
    blocks = ['con', 'walk', t_str]
    train_id = '_'.join(b for b in blocks if b != '')
    log_path = os.path.join(LOG_DIR, train_id)
    save_path = os.path.join(MODEL_DIR, f"{train_id}.pt")

    # MVAE motion npz data
    npz_path = os.path.join(NPZ_DIR, "walk1_subject5.npz")
    mvae_motion = torch.from_numpy(
        np.load(npz_path)["mvae_motion"]
    ).float().to(args.device)
    motion_len = mvae_motion.shape[0]
    frame_size = mvae_motion.shape[1]
    dt = 0.01
    halflife = 0.27

    # Init MVAE condition
    start_idx = torch.randint(
        low=0,
        high=motion_len,
        size=(args.num_parallel,)
    ).long().to(args.device)
    condition = torch.zeros(
        (args.num_parallel, frame_size)
    ).to(args.device)
    condition.copy_(mvae_motion[start_idx])

    # Init root rotation, velocity, avel
    root_rotvec = np.array([[0,1,0] for _ in range(args.num_parallel)])
    root_rot = R.from_rotvec(root_rotvec)
    root_vel = np.zeros((args.num_parallel, 3))
    root_avel = np.zeros((args.num_parallel, 3))

    # Init camera rotation and user input
    angle = np.random.uniform(-np.pi, np.pi, size=args.num_parallel)
    y_rotvec = np.array([[0,1,0] for _ in range(args.num_parallel)])
    y_rot = R.from_rotvec(angle[:, None] * y_rotvec)

    # trained MVAE model
    mvae_path = os.path.join(MODEL_DIR, "walk1_subject5_240129_190220.pt")
    mvae = torch.load(mvae_path, map_location=args.device)
    mvae.eval()

    # learning params
    lr = 1e-4
    # Init controller network
    con_mvae = ConMVAE(
        signal_size=args.signal_size,
        mvae=mvae
    ).to(args.device)
    con_mvae.train()
    con_optim = torch.optim.Adam(
        params=con_mvae.parameters(),
        lr=lr,
    )
    
    num_updates = int(
        args.num_frames / args.num_parallel / args.num_steps_per_rollout
    )
    writer = SummaryWriter(log_path)
    global_step = 0
    for i in tqdm(range(num_updates)):
        for step in range(args.num_steps_per_rollout):
            # Check if the target needs to be changed
            if (step % args.facing_switch_every) == 0:
                angle = np.random.uniform(-np.pi, np.pi, size=args.num_parallel)
                y_rotvec = np.array([[0,1,0] for _ in range(args.num_parallel)])
                y_rot = R.from_rotvec(angle[:, None] * y_rotvec)

            desired_vel = y_rot.apply(np.array([0, 0, 1]))
            desired_rot = y_rot.as_quat()

            # update root_vel
            np_condition = condition.detach().cpu().numpy()
            root_vel[:, 0] = np_condition[:, 0]
            root_vel[:, 2] = np_condition[:, 1]
            root_vel = root_rot.apply(root_vel)
            root_avel[:, 1] = np_condition[:, 2]

            # global con_vel(3) and con_avel(3)
            con_vel = SpringUtil.batch_spring_vel(root_vel, desired_vel, 0, halflife, dt)
            con_avel = SpringUtil.batch_spring_avel(root_rot.as_quat(), desired_rot, root_avel, halflife, dt)

            con_vel = root_rot.apply(desired_vel, inverse=True)[:, :2]
            con_avel = con_avel[:, 1]
            con_vel = torch.from_numpy(con_vel).float().to(args.device)
            con_avel = torch.from_numpy(con_avel).float().to(args.device)
            con_signal = torch.cat((con_vel, con_avel[:, None]), dim=-1)

            # torch.autograd.set_detect_anomaly(True)
            next_frame, loss = feed_con_mvae(con_mvae, con_signal, condition)
            
            condition.copy_(next_frame.detach())
            con_optim.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            con_optim.step()
            
            # Update root rot
            omega = next_frame[:, 2].detach().cpu().numpy()
            root_avel[:, 2] = omega
            root_quat = root_rot.as_quat()
            root_quat = QuatUtil.quat_integrate(root_quat, root_avel, dt)
            root_rot = R.from_quat(root_quat)

            with torch.no_grad():
                writer.add_scalar("direct_loss", loss, global_step)
                global_step += 1

        # Reset condition
        start_idx = torch.randint(
            low=0,
            high=motion_len,
            size=(args.num_parallel,)
        ).long().to(args.device)
        condition = torch.zeros(
            (args.num_parallel, frame_size)
        ).to(args.device)
        condition.copy_(mvae_motion[start_idx])

    torch.save(con_mvae, save_path)