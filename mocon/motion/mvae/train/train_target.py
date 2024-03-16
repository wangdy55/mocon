import torch
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os, time

from mocon.motion.mvae.model.ConMVAE import ConMVAE

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

def feed_con_mvae(con_mvae, con_signal, condition, root_info):
    next_frame = con_mvae(con_signal, condition)


    direct_loss = (
        target_info[:, 2].cos() - root_info[:, 2].cos() +
        target_info[:, 2].sin() - root_info[:, 2].sin()
    ).pow(2).mean()
    # return next_frame, con_loss

def train_target():
    args = SimpleNamespace(
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
        signal_size=2,
        num_frames=10e6,
        num_parallel=100,
        num_steps_per_rollout=1200,
        facing_switch_every=240
    )

    # path of log and checkpoint
    t_str = time.strftime("%y%m%d_%H%M%S", time.localtime(round(time.time())))
    blocks = ['target', 'walk', t_str]
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

    # Init global root info and target info
    root_info = torch.zeros(
        (args.num_parallel, 3) # (x, z, facing)
    ).to(args.device)
    target_info = torch.zeros(
        (args.num_parallel, 3)
    ).to(args.device)
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
    # writer = SummaryWriter(log_path)
    # global_step = 0
    for i in tqdm(range(num_updates)):
        for step in range(args.num_steps_per_rollout):
            # Check if the target needs to be changed
            if (step % args.facing_switch_every) == 0:
                rand_delta = torch.rand(args.num_parallel) * np.pi - np.pi/2
                rand_delta = rand_delta.to(args.device)
                target_info[:, 2].copy_(root_info[:, 2] + rand_delta)
                target_info[:, 2].remainder_(2*np.pi)
            # Update target
            target_info[:, :2].copy_(root_info[:, :2])
            con_signal = torch.cat(
                (
                    torch.cos(target_info[:, 2, None]),
                    torch.sin(target_info[:, 2, None])
                ), dim=1
            )
            # TODO: global target => local target (maybe)
            target_info[:, :2].add_(10 * con_signal)
            
            # target_direct as controller input
            torch.autograd.set_detect_anomaly(True)
            next_frame = con_mvae(con_signal, condition)

            # Integrate => root_info
            rotmat = get_2d_rotmat(root_info[:, 2, None])
            vel_xz = (rotmat * next_frame[:, :2, None]).sum(dim=-1)
            root_info[:, :2] = (vel_xz * dt) + root_info[:, :2]
            avel_y = next_frame[:, 2]
            root_info[:, 2] = ((avel_y * dt) + root_info[:, 2]).remainder_(2*np.pi)
            con_res = torch.cat(
                (
                    torch.cos(root_info[:, 2, None]),
                    torch.sin(root_info[:, 2, None])
                ), dim=1
            )

            # loss of velocity direction
            direct_loss = (con_signal - con_res).pow(2).mean()
            condition.copy_(next_frame.detach())

            con_optim.zero_grad()
            with torch.autograd.detect_anomaly():
                direct_loss.backward()
            con_optim.step()

            # with torch.no_grad():
            #     writer.add_scalar("direct_loss", direct_loss, global_step)
            #     global_step += 1

        # Reset condition
        start_idx = torch.randint(
            low=0,
            high=motion_len,
            size=(args.num_parallel)
        ).long().to(args.device)
        condition = torch.zeros(
            (args.num_parallel, frame_size)
        ).to(args.device)
        condition.copy_(mvae_motion[start_idx])

    torch.save(con_mvae, save_path)