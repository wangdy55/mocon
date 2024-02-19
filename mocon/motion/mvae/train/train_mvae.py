import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from types import SimpleNamespace
import numpy as np
import os, time, copy

from mocon.motion.mvae.train.train_util import update_linear_schedule
from mocon.motion.mvae.model.MotionVAE import MotionVAE, MotionMixtureVAE, MotionMixtureSpecialistVAE

# dir.
MOCAP_DIR = "mocon/motion/mocap/npz"
LOG_DIR = "mocon/motion/mvae/log"
MODEL_DIR = "mocon/motion/mvae/model"

# spec.
MOCAP = "run2_subject1.npz"
MODEL = "sp"
BETA = 0.5
TAIL = "" # supplemental info.

def feed_vae(mvae, ground_truth, condition, future_weights):
    condition = condition.flatten(start_dim=1, end_dim=2)
    flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)

    output_shape = (-1, mvae.num_future_predictions, mvae.frame_size)

    if isinstance(mvae, MotionMixtureSpecialistVAE):
        vae_output, mu, logvar, coefficient = mvae(flattened_truth, condition)

        recon_loss = (vae_output - ground_truth).pow(2).mean(dim=2).mul(-0.5).exp()
        recon_loss = (recon_loss * coefficient).sum(dim=1).log().mul(-1).mean()

        # Sample a next frame from experts
        indices = torch.distributions.Categorical(coefficient).sample()
        # was (expert, batch, feature), after select is (batch, feature)
        vae_output = vae_output[torch.arange(vae_output.size(0)), indices]
        vae_output = vae_output.view(output_shape)

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
        kl_loss /= logvar.numel()

        return (vae_output, mu, logvar), (recon_loss, kl_loss)

    else:
        # MotionVAE and MotionMixtureVAE
        vae_output, mu, logvar = mvae(flattened_truth, condition)
        vae_output = vae_output.view(output_shape)

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
        kl_loss /= logvar.numel()

        recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
        recon_loss = recon_loss.mul(future_weights).sum()

        return (vae_output, mu, logvar), (recon_loss, kl_loss)

def train_mvae():
    args = SimpleNamespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        mocap_file=os.path.join(MOCAP_DIR, MOCAP),
        norm_mode="zscore",
        latent_size=32,
        # num_embeddings=12,
        num_experts=6,
        num_condition_frames=1,
        num_future_predictions=1,
        num_steps_per_rollout=8,
        kl_beta=BETA,
        load_saved_model=False,
    )

    # learning params
    teacher_epochs = 20
    ramping_epochs = 20
    student_epochs = 100
    args.num_epochs = teacher_epochs + ramping_epochs + student_epochs
    args.mini_batch_size = 256
    args.initial_lr = 1e-4
    args.final_lr = 1e-7

    raw_data = np.load(args.mocap_file)["mvae_motion"]
    mocap_data = torch.from_numpy(raw_data).float().to(args.device)

    max = mocap_data.max(dim=0)[0]
    min = mocap_data.min(dim=0)[0]
    avg = mocap_data.mean(dim=0)
    std = mocap_data.std(dim=0)

    # avg[-(num_not_ee-1)*6:], std[-(num_not_ee-1)*6:] = 0, 1
    # std[torch.abs(std) < 0.1] = 1.0

    # Make sure we don't divide by 0
    std[std == 0] = 1.0

    normalization = {
        "mode": args.norm_mode,
        "max": max,
        "min": min,
        "avg": avg,
        "std": std,
    }

    if args.norm_mode == "minmax":
        mocap_data = 2 * (mocap_data - min) / (max - min) - 1

    elif args.norm_mode == "zscore":
        mocap_data = (mocap_data - avg) / std

    batch_size = mocap_data.size()[0]
    frame_size = mocap_data.size()[1]

    selectable_indices = np.arange(batch_size - args.num_steps_per_rollout
                                   - (args.num_condition_frames - 1)
                                   - (args.num_future_predictions - 1))

    mvae = MotionMixtureSpecialistVAE(
        frame_size,
        args.latent_size,
        args.num_condition_frames,
        args.num_future_predictions,
        normalization,
        args.num_experts,
    ).to(args.device)

    if isinstance(mvae, MotionVAE):
        model_args = "c{}_l{}_b{}".format(
            args.num_condition_frames, args.latent_size, args.kl_beta
        )
    elif isinstance(mvae, MotionMixtureVAE):
        model_args = "c{}_e{}_l{}_b{}".format(
            args.num_condition_frames, args.num_experts, args.latent_size, args.kl_beta
        )
    elif isinstance(mvae, MotionMixtureSpecialistVAE):
        model_args = "c{}_s{}_l{}_b{}".format(
            args.num_condition_frames, args.num_experts, args.latent_size, args.kl_beta
        )

    # path of checkpoint and log
    t_str = time.strftime("%y%m%d_%H%M%S", time.localtime(round(time.time())))
    blocks = [t_str, MODEL, model_args, TAIL]
    train_id = '_'.join(b for b in blocks if b != '')
    log_path = os.path.join(LOG_DIR, train_id)
    save_path = os.path.join(
        MODEL_DIR,
        f"{MOCAP.split('.')[0]}_{t_str}.pt"
    )

    if args.load_saved_model:
        mvae = torch.load(save_path, map_location=args.device)

    mvae.train()

    vae_optimizer = optim.Adam(mvae.parameters(), lr=args.initial_lr)

    sample_schedule = torch.cat(
        (
            # First part is pure teacher forcing
            torch.zeros(teacher_epochs),
            # Second part with schedule sampling
            torch.linspace(0.0, 1.0, ramping_epochs),
            # Last part is pure student
            torch.ones(student_epochs),
        )
    )

    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )

    # buffer for later
    training_shape = (args.mini_batch_size, args.num_condition_frames, frame_size)
    history = torch.empty(training_shape).to(args.device) # rewrite in each epoch

    # TensorBoard log
    writer = SummaryWriter(log_path)
    global_step = 0

    for e in tqdm(range(1, args.num_epochs + 1)):
        # Sampler[sampler1, sampler2, ...]
        sampler = BatchSampler(
            SubsetRandomSampler(selectable_indices), # no-repeat random sampling
            args.mini_batch_size,
            drop_last=True,
        )

        update_linear_schedule(
            vae_optimizer, e-1, args.num_epochs, args.initial_lr, args.final_lr
        )
      
        for indices in sampler:
            t_indices = torch.LongTensor(indices) # randomly sampled indices

            # Condition is from newest...oldest, i.e. (t-1, t-2, ... t-n)
            condition_range = (
                t_indices.repeat((args.num_condition_frames, 1)).t()
                + torch.arange(args.num_condition_frames - 1, -1, -1).long()
            )

            t_indices += args.num_condition_frames
            history[:, : args.num_condition_frames].copy_(mocap_data[condition_range])

            for offset in range(args.num_steps_per_rollout):
                use_student = torch.rand(1) < sample_schedule[e - 1]

                prediction_range = (
                    t_indices.repeat((args.num_future_predictions, 1)).t()
                    + torch.arange(offset, offset + args.num_future_predictions).long()
                )
                ground_truth = mocap_data[prediction_range]
                condition = history[:, : args.num_condition_frames]

                # A step of the VAE
                (vae_output, _, _), (recon_loss, kl_loss) = feed_vae(
                    mvae, ground_truth, condition, future_weights
                )

                # Orthognalize vec6d
                # vae_output[..., -19*6:] = batch_orth_vec6d(vae_output[..., -19*6:]).detach()

                history = history.roll(1, dims=1)
                next_frame = vae_output[:, 0] if use_student else ground_truth[:, 0]
                history[:, 0].copy_(next_frame.detach())

                vae_optimizer.zero_grad()
                total_loss = recon_loss + args.kl_beta * kl_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(mvae.parameters(), 10)
                vae_optimizer.step()

                with torch.no_grad():
                    writer.add_scalar("kl_loss", kl_loss, global_step)
                    writer.add_scalar("recon_loss", recon_loss, global_step)
                    writer.add_scalar("total_loss", total_loss, global_step)
                    global_step += 1

        torch.save(mvae, save_path)
