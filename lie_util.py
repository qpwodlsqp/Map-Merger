import torch
import numpy as np
'''
Original MyMap Dataset input just uses parameters directly from Lie group
'''
def target_to_algebra(pose):
    # pose target to Lie algebra (N, 3) => (N, 3)
    # this is available in anywhere as long as the input alg is batched (N, 3)
    # Before the transformation, always do value scaling!
    N = pose.shape[0]
    t = pose[:, :2] * 384 / 128 / 20   # (N, 2)
    theta = pose[:, 2:] * np.pi / 180  # (N, 1)
    A = torch.sin(theta + 1e-9) / (theta + 1e-9) # avoid zero division (N, 1)
    B = (1 - torch.cos(theta + 1e-9)) / (theta + 1e-9) # (N, 1)
    Vinv = torch.cat([A, B, -B, A], dim=-1).reshape(N, 2, 2) / (A*A + B*B).reshape(N, 1, 1)
    u = torch.bmm(Vinv, t.unsqueeze(-1)).squeeze(-1) # (N, 2)
    return torch.cat([u, theta], dim=-1)

def algebra_to_target(alg):
    # alg back to target (N, 3) => (N, 3)
    # this is available in anywhere as long as the input alg is batched (N, 3)
    # After the transformation, always do value scaling!
    N = alg.shape[0]
    SE2 = exp_map(alg)
    t = SE2[:, :2, 2:].reshape(N, 2) * 20 * 128 / 384 # (N, 2)
    theta = alg[:, 2:] * 180 / np.pi                  # (N, 1)
    return torch.cat([t, theta], dim=-1) # (N, 3)

def exp_map(alg):
    # Lie algebra to Lie group SE(2)
    # alg \in (N, 3), pose = (tw, th, theta)
    N = alg.shape[0]
    u = alg[:, :2].reshape(N, 2, 1)
    theta = alg[:, 2:]
    so2 = torch.cat([torch.zeros_like(theta), -theta, theta, torch.zeros_like(theta)], dim=-1).reshape(N, 2, 2)
    zeros = torch.cat([torch.zeros_like(theta), torch.zeros_like(theta), torch.zeros_like(theta)], dim=-1)
    se2 = torch.cat([torch.cat([so2, u], dim=-1), zeros.unsqueeze(1)], dim=1)
    return torch.linalg.matrix_exp(se2)

def log_map(group):
    # Lie group to Lie algebra se(2)
    # group \in (N, 3, 3)
    N = group.shape[0]
    cos = group[:, 0, 0].unsqueeze(-1)
    sin = group[:, 1, 0].unsqueeze(-1)
    theta = torch.atan2(sin, cos)
    A = torch.sin(theta+1e-9) / (theta + 1e-9)
    B = (1 - torch.cos(theta+1e-9)) / (theta + 1e-9)
    Vinv = torch.cat([A, B, -B, A], dim=-1).reshape(N, 2, 2) / (A*A + B*B).reshape(N, 1, 1)
    t = group[:, :2, 2:] # (N, 2, 1)
    u = torch.bmm(Vinv, t).reshape(N, 2)
    return torch.cat([u, theta], dim=-1)

def lie_loss(pred, gt, cov):
    # Assume cov is already sent to device
    # Assume pred and gt follows Lie algebra se(2) with batch
    assert pred.shape == gt.shape
    N = pred.shape[0]
    diff = log_map(torch.bmm(exp_map(pred), torch.linalg.inv(exp_map(gt)))) # (N, 3)
    loss = 0.
    for i in range(N):
        diff_point = diff[i:i+1, :] # (1, 3)
        loss += diff_point @ torch.linalg.inv(cov.to(pred)) @ diff_point.t()
    return loss / N


