import torch
import numpy as np

def add_gaussian_white_noise(tensor, mean=0.0, std=1.0, rd=1):
    noise = torch.normal(mean, std * rd, tensor.size())
    return tensor + noise

def add_red_noise(tensor, alpha=0.5, rd=1):
    alpha = alpha * rd
    batch_size, seq_len, feature_num = tensor.size()
    red_noise = torch.zeros_like(tensor)
    
    red_noise[:, 0, :] = torch.normal(0, 1, (batch_size, feature_num))

    for i in range(1, seq_len):
        red_noise[:, i, :] = alpha * red_noise[:, i-1, :] + torch.normal(0, (1 - alpha**2)**0.5, (batch_size, feature_num))
    
    return tensor + red_noise

def phase_shift(tensor, shift=None, rd=1):
    batchsize, seqlen, featurenum = tensor.shape
    shift = int(rd * seqlen * 0.2)
    if shift >= 0:
        shifted_tensor = torch.cat([tensor[:, shift:, :], tensor[:, :shift, :]], dim=1)
    else:
        shift = abs(shift)
        shifted_tensor = torch.cat([tensor[:, seqlen-shift:, :], tensor[:, :seqlen-shift, :]], dim=1)
    return shifted_tensor

def period_scale(tensor, scale=2.0, rd=1):
    batchsize, seqlen, featurenum = tensor.shape

    new_seqlen = int(seqlen * scale * rd)  # 使用 rd 调整缩放比例
    
    # 创建时间轴的坐标范围为 0 到 1，以保持周期性
    time_axis = torch.linspace(0, 1, steps=seqlen, device=tensor.device, dtype=tensor.dtype)
    new_time_axis = torch.linspace(0, 1, steps=new_seqlen, device=tensor.device, dtype=tensor.dtype)
    
    # 为每个特征创建网格
    y_axis = torch.arange(featurenum, device=tensor.device, dtype=tensor.dtype)
    grid = torch.stack(torch.meshgrid(time_axis, y_axis, indexing='ij'), dim=-1)  # (seqlen, featurenum, 2)
    grid = grid.unsqueeze(0).repeat(batchsize, 1, 1, 1)  # (batchsize, seqlen, featurenum, 2)
    
    # 调整网格到新的序列长度
    new_grid = torch.stack(torch.meshgrid(new_time_axis, y_axis, indexing='ij'), dim=-1)  # (new_seqlen, featurenum, 2)
    new_grid = new_grid.unsqueeze(0).repeat(batchsize, 1, 1, 1)  # (batchsize, new_seqlen, featurenum, 2)
    
    # 使用 grid_sample 进行采样
    tensor = tensor.unsqueeze(1)  # (batchsize, 1, seqlen, featurenum)
    tensor = torch.nn.functional.grid_sample(tensor, new_grid, mode='bilinear', align_corners=True).squeeze(1)  # (batchsize, new_seqlen, featurenum)
    
    if new_seqlen > seqlen:
        tensor = tensor[:, :seqlen, :]
    else:
        pad_size = seqlen - new_seqlen
        tensor = torch.cat([tensor, tensor[:, :pad_size, :]], dim=1)

    return tensor

def global_mean_shift(tensor, shift_amount=1, rd=1):
    if isinstance(shift_amount, (int, float)):
        shift_amount = tensor.new_full((1, 1, tensor.size(2)), shift_amount)
    elif isinstance(shift_amount, torch.Tensor):
        assert shift_amount.size(0) == tensor.size(2), "the length of shift_amount must be the same as feature_num"
    else:
        raise ValueError("shift_amount must be a scalar or a vector of the same length as feature_num")
    
    return tensor + shift_amount * rd

def data_range_scaling(tensor, scale_factor=0.9, rd=1):
    if isinstance(scale_factor, (int, float)):
        scale_factor = tensor.new_full((1, 1, tensor.size(2)), scale_factor)
    elif isinstance(scale_factor, torch.Tensor):
        assert scale_factor.size(0) == tensor.size(2), "the length of scale_factor must be the same as feature_num"
    else:
        raise ValueError("scale_factor must be a scalar or a vector of the same length as feature_num")
    
    return tensor * scale_factor * rd

def add_missing_values(tensor, missing_rate=0.05, rd=1):
    mask = torch.rand(tensor.size()) < missing_rate * rd
    tensor[mask] = 0.0
    return tensor

def add_outlier_values(tensor, outlier_rate=0.05, magnitude=20, rd=1):
    mask = torch.rand(tensor.size()) < outlier_rate * rd
    # 生成异常值，并确保异常值的类型与 tensor 相同
    outliers = magnitude * torch.randn(*tensor.size()).to(tensor.dtype)
    tensor[mask] = outliers[mask]
    return tensor

def adaptivity(x, i, args, rd=1):
    adp_type = ""
    if i % 2 == 0:
        if args.noiseness:
            trans_x = add_gaussian_white_noise(x, rd)
            adp_type = "noiseness"
        if args.periodicity:
            trans_x = phase_shift(x, rd)
            adp_type = "periodicity"
        if args.distribution:
            trans_x = global_mean_shift(x, rd)
            adp_type = "distribution"
        if args.anomaly:
            trans_x = add_missing_values(x, rd)
            adp_type = "anomaly"
    else:
        if args.noiseness:
            trans_x = add_red_noise(x, rd)
            adp_type = "noiseness"
        if args.periodicity:
            trans_x = period_scale(x, rd)
            adp_type = "periodicity"
        if args.distribution:
            trans_x = data_range_scaling(x, rd)
            adp_type = "distribution"
        if args.anomaly:
            trans_x = add_outlier_values(x, rd)
            adp_type = "anomaly"
    return trans_x, adp_type