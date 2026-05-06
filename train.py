"""
Behavioral Cloning: учим нейросеть имитировать rule-based policy.

Usage:
    conda activate base
    python train.py [--data data/episodes.ndjson] [--epochs 40] [--batch 512]
                   [--out data/model.pt] [--onnx data/model.onnx]
                   [--hidden 512] [--lr 3e-4] [--val-split 0.1]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OBS_DIM    = 455
ACTION_DIM = 45
MAX_DRONES = 15

DRONE_ALIVE_IDX = [i * 13 + 1 for i in range(MAX_DRONES)]  # alive flag per drone slot in obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',      default='data/episodes.ndjson')
    ap.add_argument('--epochs',    type=int,   default=40)
    ap.add_argument('--batch',     type=int,   default=512)
    ap.add_argument('--lr',        type=float, default=3e-4)
    ap.add_argument('--hidden',    type=int,   default=512)
    ap.add_argument('--val-split', type=float, default=0.1)
    ap.add_argument('--out',       default='data/model.pt')
    ap.add_argument('--onnx',      default='data/model.onnx')
    ap.add_argument('--log',       default='data/train_log.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  PyTorch {torch.__version__}')

    obs_np, act_np = load_data(args.data)

    n      = len(obs_np)
    n_val  = max(1, int(n * args.val_split))
    idx    = np.random.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    obs_t = torch.from_numpy(obs_np)
    act_t = torch.from_numpy(act_np)

    train_dl = DataLoader(TensorDataset(obs_t[train_idx], act_t[train_idx]),
                          batch_size=args.batch, shuffle=True,  pin_memory=device.type == 'cuda')
    val_dl   = DataLoader(TensorDataset(obs_t[val_idx],   act_t[val_idx]),
                          batch_size=args.batch, shuffle=False, pin_memory=device.type == 'cuda')

    model = TeamPolicyNet(hidden=args.hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params:,} params  |  train {len(train_idx):,}  val {len(val_idx):,} steps')
    print(f'{"Epoch":>6}  {"train":>8}  {"val":>8}  {"lr":>8}  {"time":>6}')
    print('─' * 50)

    log       = []
    best_val  = float('inf')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = run_epoch(model, train_dl, device, opt)

        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_dl, device, opt=None)

        sched.step()
        elapsed = time.time() - t0
        lr_now  = sched.get_last_lr()[0]

        print(f'{epoch:6d}  {train_loss:8.5f}  {val_loss:8.5f}  {lr_now:8.2e}  {elapsed:5.1f}s')

        log.append({'epoch': epoch, 'train_loss': round(train_loss, 6), 'val_loss': round(val_loss, 6)})
        with open(args.log, 'w') as f:
            json.dump(log, f)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(),
                        'obs_dim': OBS_DIM, 'action_dim': ACTION_DIM,
                        'hidden': args.hidden, 'epoch': epoch}, args.out)

    print(f'\nBest val loss: {best_val:.5f}  →  {args.out}')

    # Reload best checkpoint and export ONNX
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    export_onnx(model, args.onnx, device)


def run_epoch(model, loader, device, opt):
    total_loss = 0.0
    total_w    = 0.0
    for obs_b, act_b in loader:
        obs_b = obs_b.to(device, non_blocking=True)
        act_b = act_b.to(device, non_blocking=True)

        alive  = obs_b[:, DRONE_ALIVE_IDX]                   # (B, 15)  — 1 if drone slot active
        mask   = alive.unsqueeze(-1).expand(-1, -1, 3)       # (B, 15, 3)
        mask   = mask.reshape(obs_b.shape[0], ACTION_DIM)    # (B, 45)

        pred   = model(obs_b)
        loss   = ((pred - act_b) ** 2 * mask).sum()
        w      = mask.sum().clamp(min=1)

        if opt is not None:
            opt.zero_grad()
            (loss / w).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total_loss += loss.item()
        total_w    += w.item()

    return total_loss / max(total_w, 1.0)


def export_onnx(model, path, device):
    dummy = torch.zeros(1, OBS_DIM, device=device)
    torch.onnx.export(
        model, dummy, path,
        input_names=['obs'], output_names=['action'],
        dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}},
        opset_version=17,
    )
    print(f'ONNX exported → {path}')


def load_data(path):
    print(f'Loading {path} ...')
    obs_list, act_list = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            for step in ep['steps']:
                obs_list.append(step['obs'])
                act_list.append(step['action'])
    obs = np.array(obs_list, dtype=np.float32)
    act = np.array(act_list, dtype=np.float32)
    print(f'Loaded {len(obs):,} steps')
    return obs, act


class TeamPolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    main()
