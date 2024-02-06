def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, final_lr=0):
    # Decreases the learning rate linearly
    lr = initial_lr - (initial_lr - final_lr) * epoch / float(total_num_epochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr