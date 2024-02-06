import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os, time

from mocon.motion.MoconTask import MoconTask
from mocon.motion.mvae.model.MotionVAEController import MotionVAEController

LOG_DIR = "mocon/motion/mvae/log"
MODEL_DIR = "mocon/motion/mvae/model"
TAIL = "" 

def main():
    # sampling parameters
    num_frames = 10e5
    num_parallel=100
    num_steps_per_rollout = 1200
    num_updates = int(
        num_frames / num_parallel / num_steps_per_rollout
    )

    # learning parameters
    lr = 3e-5
    # mini_batch_size = 1000
    # num_mini_batch = (
    #     num_parallel * num_steps_per_rollout // mini_batch_size
    # )

    mocon_task = MoconTask(
        num_parallel=100,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        npz_path="mocon/motion/mocap/npz/walk1_subject5.npz",
        mvae_path="mocon/motion/mvae/model/walk1_subject5_240129_190220.pt",
    )
    mocon_task.reset()

    controller = MotionVAEController(
        input_size=mocon_task.ctrl_condition_size,
        output_size=mocon_task.action_size,
    ).to(mocon_task.device)

    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=lr,
    )

    # path of checkpoint and log
    t_str = time.strftime("%y%m%d_%H%M%S", time.localtime(round(time.time())))
    log_path = os.path.join(
        LOG_DIR,
        f"{t_str}_con_walk1_subject5"
    )
    save_path = os.path.join(
        MODEL_DIR,
        f"con_walk1_subject5_{t_str}.pt"
    )

    # TensorBoard log
    writer = SummaryWriter(log_path)
    global_step = 0

    for update in tqdm(range(num_updates)):
        for step in range(num_steps_per_rollout):
            ctrl_input = mocon_task.get_controller_input()
            action = controller(ctrl_input)
            mocon_task.update(action)

            controller_optimizer.zero_grad()
            loss = (
                mocon_task.target_direct 
                - mocon_task.root_facing
            ).pow(2).mean()
            loss.requires_grad_(True)
            loss.backward()
            controller_optimizer.step()

            with torch.no_grad():
                writer.add_scalar("loss", loss, global_step)
                global_step += 1

            mocon_task.reset_target()

        mocon_task.reset()

    torch.save(controller.state_dict(), save_path)

if __name__ == "__main__":
    main()