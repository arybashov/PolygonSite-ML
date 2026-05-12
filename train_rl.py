"""
PPO training for the drone-interception team policy.

Usage:
    python train_rl.py [--updates 2000] [--rollout 2048] [--envs 8]
                       [--epochs 10] [--lr 3e-4] [--clip 0.2]
                       [--ndrones 5] [--nanti 5]
                       [--eval-every 50] [--eval-eps 50]
                       [--out results.json] [--label "PPO"]
                       [--resume data/model_rl.pt]
"""
import argparse
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from py.sim import Simulation, OBS_DIM, ACTION_DIM, DEFAULT_PARAMS

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--updates',    type=int,   default=2000)
    p.add_argument('--rollout',    type=int,   default=1024,  help='steps per env per update')
    p.add_argument('--envs',       type=int,   default=8,     help='parallel envs')
    p.add_argument('--epochs',     type=int,   default=10,    help='PPO update epochs')
    p.add_argument('--minibatch',  type=int,   default=256)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--clip',       type=float, default=0.1)
    p.add_argument('--gamma',      type=float, default=1.0)
    p.add_argument('--lam',        type=float, default=1.0)
    p.add_argument('--ent-coef',   type=float, default=0.0)
    p.add_argument('--val-coef',   type=float, default=0.5)
    p.add_argument('--max-grad',   type=float, default=0.5)
    p.add_argument('--target-kl',  type=float, default=0.1,   help='KL early-stop threshold per epoch')
    p.add_argument('--anneal-lr',  action='store_true', default=True, help='linear LR annealing')
    p.add_argument('--ndrones',    type=int,   default=None)
    p.add_argument('--nanti',      type=int,   default=None)
    p.add_argument('--eval-every', type=int,   default=20)
    p.add_argument('--eval-eps',   type=int,   default=20)
    p.add_argument('--out',        type=str,   default='results.json')
    p.add_argument('--label',      type=str,   default='PPO')
    p.add_argument('--resume',     type=str,   default=None)
    p.add_argument('--log',        type=str,   default='data/train_log.json')
    return p.parse_args()

# ── Network ──────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACTION_DIM, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden // 2, act_dim),
            nn.Tanh(),
        )
        # fixed std — not learned; prevents entropy drift with sparse rewards
        self.register_buffer('log_std', torch.full((act_dim,), -0.5))
        self.critic = nn.Linear(hidden // 2, 1)

        # orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        feat = self.backbone(obs)
        mean = self.actor_mean(feat)
        std  = self.log_std.exp().expand_as(mean)
        val  = self.critic(feat).squeeze(-1)
        return mean, std, val

    def get_action(self, obs, deterministic=False):
        mean, std, val = self.forward(obs)
        dist   = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        logp   = dist.log_prob(action.clamp(-1 + 1e-6, 1 - 1e-6)).sum(-1)
        action = action.clamp(-1.0, 1.0)
        return action, logp, val

    def evaluate(self, obs, action):
        mean, std, val = self.forward(obs)
        dist    = Normal(mean, std)
        logp    = dist.log_prob(action.clamp(-1 + 1e-6, 1 - 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, val

# ── Vectorised envs (CPU, pure Python) ───────────────────────────────────────

class VecEnv:
    def __init__(self, n_envs, params):
        self.n_envs = n_envs
        self.params = params
        self.sims   = [Simulation(params) for _ in range(n_envs)]
        self.seed_counter = 0

    def reset_all(self):
        obs = []
        for sim in self.sims:
            obs.append(sim.reset(self.seed_counter))
            self.seed_counter += 1
        return np.stack(obs)          # (n_envs, OBS_DIM)

    def step(self, actions):
        """actions: (n_envs, ACTION_DIM) numpy array"""
        obs_list, rew_list, done_list = [], [], []
        for i, sim in enumerate(self.sims):
            obs, rew, done, _ = sim.step(actions[i])
            if done:
                obs = sim.reset(self.seed_counter)
                self.seed_counter += 1
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
        )

# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, obs_dim, act_dim, device):
        T, E = rollout_steps, n_envs
        self.obs     = torch.zeros(T, E, obs_dim,  device=device)
        self.actions = torch.zeros(T, E, act_dim,  device=device)
        self.logps   = torch.zeros(T, E,           device=device)
        self.rewards = torch.zeros(T, E,           device=device)
        self.dones   = torch.zeros(T, E,           device=device)
        self.values  = torch.zeros(T, E,           device=device)
        self.ptr     = 0
        self.T       = T

    def store(self, obs, actions, logps, rewards, dones, values):
        t = self.ptr
        self.obs[t]     = obs
        self.actions[t] = actions
        self.logps[t]   = logps
        self.rewards[t] = rewards
        self.dones[t]   = dones
        self.values[t]  = values
        self.ptr += 1

    def compute_returns(self, last_values, gamma, lam):
        """GAE-λ advantage estimation."""
        T, E = self.T, self.rewards.shape[1]
        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros(E, device=self.rewards.device)
        for t in reversed(range(T)):
            next_val  = last_values if t == T - 1 else self.values[t + 1]
            next_done = self.dones[t]
            delta = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            gae   = delta + gamma * lam * (1 - next_done) * gae
            advantages[t] = gae
        returns = advantages + self.values
        return advantages, returns

    def flatten(self):
        T, E = self.T, self.obs.shape[1]
        def f(x): return x.reshape(T * E, *x.shape[2:])
        return f(self.obs), f(self.actions), f(self.logps), f(self.rewards), f(self.dones), f(self.values)

    def reset(self):
        self.ptr = 0

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, params, n_episodes, device, seed_start=100_000):
    model.eval()
    records = []
    max_steps = 40000  # > 500s × 60Hz = 30000 physics steps
    sim = Simulation(params)
    with torch.no_grad():
        for ep in range(n_episodes):
            obs = sim.reset(seed_start + ep)
            total_reward = 0.0
            for _ in range(max_steps):
                t_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = model.get_action(t_obs, deterministic=True)
                obs, rew, done, _ = sim.step(np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0))
                total_reward += rew
                if done:
                    break
            hits        = sum(1 for d in sim.drones if d['mode'] == 'hit')
            intercepted = sum(1 for d in sim.drones if d['mode'] == 'intercepted')
            total       = len(sim.drones)
            timed_out   = not all(not d['alive'] for d in sim.drones)
            records.append({
                'seed': seed_start + ep,
                'reward': total_reward,
                'hits': hits,
                'intercepted': intercepted,
                'total': total,
                'time': round(sim.sim_time),
                'timedOut': timed_out,
            })
    model.train()
    n = len(records)
    def avg(k):    return sum(r[k] for r in records) / n
    def ratio(a,b): return sum(r[a]/r[b] for r in records) / n
    summary = {
        'mean_reward':    round(avg('reward'), 3),
        'hit_rate':       round(ratio('hits', 'total'), 3),
        'intercept_rate': round(ratio('intercepted', 'total'), 3),
        'mean_time':      round(avg('time'), 3),
        'timeout_rate':   round(sum(1 for r in records if r['timedOut']) / n, 3),
    }
    return summary, records

# ── Log helpers ───────────────────────────────────────────────────────────────

def append_train_log(path, entry):
    data = []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        pass
    data.append(entry)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

def append_results(path, label, params, summary, records, update, elapsed_sec=None):
    run = {
        'id':          int(time.time() * 1000),
        'timestamp':   datetime.utcnow().isoformat() + 'Z',
        'label':       f'{label} (update {update})',
        'policy':      'rl-onnx',
        'params':      {'ndrones': params['ndrones'], 'nanti': params['nanti']},
        'summary':     summary,
        'records':     records,
        'train_update': update,
        'elapsed_sec': round(elapsed_sec) if elapsed_sec is not None else None,
    }
    existing = []
    try:
        with open(path) as f:
            existing = json.load(f)
    except Exception:
        pass
    existing.append(run)
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)

def export_onnx(model, device, path_pt, path_onnx):
    torch.save({'model_state': model.state_dict(), 'hidden': model.hidden}, path_pt)
    model.eval()
    dummy = torch.zeros(1, OBS_DIM, device=device)
    # dynamo=False → legacy TorchScript path: ~0.15s vs ~4.5min for dynamo path
    torch.onnx.export(
        _OnnxWrapper(model), dummy, path_onnx,
        input_names=['obs'], output_names=['action'],
        dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}},
        opset_version=17,
        dynamo=False,
    )
    model.train()

class _OnnxWrapper(nn.Module):
    """Exports only the deterministic actor (mean of Gaussian)."""
    def __init__(self, ac): super().__init__(); self.ac = ac
    def forward(self, obs):
        feat = self.ac.backbone(obs)
        return self.ac.actor_mean(feat)

# ── Main training loop ────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    params = {**DEFAULT_PARAMS}
    if args.ndrones is not None: params['ndrones'] = args.ndrones
    if args.nanti   is not None: params['nanti']   = args.nanti

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    start_update = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_update = ckpt.get('update', 0)
        print(f'Resumed from {args.resume} at update {start_update}')

    vec_env = VecEnv(args.envs, params)
    buffer  = RolloutBuffer(args.rollout, args.envs, OBS_DIM, ACTION_DIM, device)

    obs_np = vec_env.reset_all()
    obs    = torch.tensor(obs_np, dtype=torch.float32, device=device)

    best_reward = -1e9
    t0 = time.time()
    train_start = t0
    eval_overhead = 0.0  # seconds spent in eval/export, excluded from elapsed_sec

    total_updates = args.updates - start_update

    for update in range(start_update, args.updates):
        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (update - start_update) / total_updates
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * frac
        # ── Collect rollout ──
        model.eval()
        buffer.reset()
        with torch.no_grad():
            for _ in range(args.rollout):
                action, logp, value = model.get_action(obs)
                actions_np = action.cpu().numpy()
                obs_np, rew_np, done_np = vec_env.step(np.clip(actions_np, -1.0, 1.0))
                buffer.store(
                    obs,
                    action,
                    logp,
                    torch.tensor(rew_np, dtype=torch.float32, device=device),
                    torch.tensor(done_np, dtype=torch.float32, device=device),
                    value,
                )
                obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

            # bootstrap final value
            _, _, last_values = model.get_action(obs)

        advantages, returns = buffer.compute_returns(last_values, args.gamma, args.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO update ──
        model.train()
        b_obs, b_act, b_logp, _, _, b_val_old = buffer.flatten()
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        total_steps = args.rollout * args.envs
        clip_losses, val_losses, ent_losses = [], [], []
        kl_early_stop = False

        for epoch_i in range(args.epochs):
            if kl_early_stop:
                break
            idx = torch.randperm(total_steps, device=device)
            for start in range(0, total_steps, args.minibatch):
                mb = idx[start:start + args.minibatch]
                new_logp, entropy, new_val = model.evaluate(b_obs[mb], b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                adv   = b_adv[mb]

                # policy clip loss
                clip_loss = -torch.min(
                    ratio * adv,
                    ratio.clamp(1 - args.clip, 1 + args.clip) * adv
                ).mean()

                # value clip loss — prevents critic from making huge jumps
                val_pred_clipped = b_val_old[mb] + (new_val - b_val_old[mb]).clamp(
                    -args.clip, args.clip)
                val_loss = 0.5 * torch.max(
                    (new_val - b_ret[mb]).pow(2),
                    (val_pred_clipped - b_ret[mb]).pow(2),
                ).mean()

                ent_loss = -entropy.mean()

                loss = clip_loss + args.val_coef * val_loss + args.ent_coef * ent_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                optimizer.step()

                clip_losses.append(clip_loss.item())
                val_losses.append(val_loss.item())
                ent_losses.append(ent_loss.item())

            # KL early stop: approximate KL after each epoch
            with torch.no_grad():
                _, _, _ = model.evaluate(b_obs, b_act)  # warm up
                new_logp_all, _, _ = model.evaluate(b_obs, b_act)
                approx_kl = ((b_logp - new_logp_all).exp() - 1 - (b_logp - new_logp_all)).mean()
                if approx_kl > args.target_kl:
                    kl_early_stop = True

        mean_clip = sum(clip_losses) / len(clip_losses)
        mean_val  = sum(val_losses)  / len(val_losses)
        mean_ent  = sum(ent_losses)  / len(ent_losses)
        now = time.time()
        fps = int(total_steps / max(now - t0, 1e-6))
        t0  = now

        log_entry = {
            'epoch':      update + 1,
            'train_loss': round(mean_clip + args.val_coef * mean_val, 6),
            'val_loss':   round(mean_val, 6),
            'entropy':    round(-mean_ent, 6),
            'clip_loss':  round(mean_clip, 6),
            'kl':         round(approx_kl.item(), 6),
        }

        if (update + 1) % 10 == 0 or update == start_update:
            stop_mark = ' [KL stop]' if kl_early_stop else ''
            print(f'update {update+1}/{args.updates}  '
                  f'clip={mean_clip:.4f}  val={mean_val:.4f}  '
                  f'ent={-mean_ent:.3f}  kl={approx_kl.item():.4f}  '
                  f'fps={fps}{stop_mark}')

        # ── Evaluation ──
        if (update + 1) % args.eval_every == 0 or update + 1 == args.updates:
            t_eval = time.time()
            summary, records = evaluate(model, params, args.eval_eps, device)
            print(f'  eval: reward={summary["mean_reward"]}  '
                  f'hit={summary["hit_rate"]}  '
                  f'intercept={summary["intercept_rate"]}')
            log_entry['eval_reward']    = summary['mean_reward']
            log_entry['eval_hit']       = summary['hit_rate']
            log_entry['eval_intercept'] = summary['intercept_rate']

            train_elapsed = time.time() - train_start - eval_overhead
            if args.out:
                append_results(args.out, args.label, params, summary, records, update + 1,
                               elapsed_sec=train_elapsed)

            # save checkpoint
            os.makedirs('data', exist_ok=True)
            ckpt_path = 'data/model_rl.pt'
            torch.save({'model_state': model.state_dict(), 'hidden': model.hidden, 'update': update + 1}, ckpt_path)

            if summary['mean_reward'] > best_reward:
                best_reward = summary['mean_reward']
                torch.save({'model_state': model.state_dict(), 'hidden': model.hidden, 'update': update + 1}, 'data/model_rl_best.pt')
                try:
                    export_onnx(model, device, 'data/model_rl_best_full.pt', 'data/model_web.onnx')
                    print(f'  exported model_web.onnx (best reward={best_reward:.3f})')
                except Exception as e:
                    print(f'  ONNX export failed: {e}')

            eval_overhead += time.time() - t_eval

        append_train_log(args.log, log_entry)

    print('Training done.')

if __name__ == '__main__':
    main()
