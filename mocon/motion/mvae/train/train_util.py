import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, final_lr=0):
    # Decreases the learning rate linearly
    lr = initial_lr - (initial_lr - final_lr) * epoch / float(total_num_epochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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