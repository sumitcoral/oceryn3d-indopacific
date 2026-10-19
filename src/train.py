#!/usr/bin/env python
"""
Stage 6: training. One-day steps (dataset.horizon 1, config/default.yaml),
then rollout fine-tuning (horizon 5, config/rollout5.yaml, which starts from
the one-day model's best.pt through train.init_from).

    python src/train.py --config config/default.yaml [--smoke [N]] [--no-resume]

Per batch: the model is applied `horizon` times, each step's re-masked
forecast feeding the next (rollout.rollout); the loss is the lead-weighted
sum of the per-step StateLoss. Gradients flow through the whole rollout.

Validation (a fixed sample of val windows, patches) reports per lead day and
variable the pooled MSE and the skill ratio model/persistence (< 1 beats
persistence, which is the day-t state held for every lead). metrics.csv gets
one row per epoch; last.pt every epoch (auto-resume), best.pt on the lowest
val loss. Checkpoints embed the merged config, so evaluate.py rebuilds the
model from ck["cfg"].

Structure, CUDA guard, warmup + cosine LR, AMP and the progress line follow
Oceryn3D Global's train.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg                                       # noqa: E402
from dataset import WindowStream, projected_ram_gb            # noqa: E402
from grid import depth_encoding                               # noqa: E402
from losses import StateLoss, lead_weights                    # noqa: E402
from model import build_model_from_cfg                        # noqa: E402
from rollout import StepAssembler, rollout                    # noqa: E402


def lr_at(step, warmup, base_lr, total):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def to_dev(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]


def run_batch(model, cfg, batch, depth_enc, loss_fn, lw, amp, device, horizon):
    """Forward rollout on one batch -> (loss, per-lead parts, preds, w, y, state0)."""
    x0, frc, fns, y, st, doy, m, e = batch
    cos_lat = st[:, 3:4]                         # static order: land bathy sin cos ...
    asm = StepAssembler(cfg, depth_enc, st, m)
    state0 = asm.remask(x0[:, :cfg.out_ch])
    w = loss_fn.weights(m, cos_lat, e)
    with torch.autocast(device_type=device.type, enabled=amp):
        preds = rollout(model, asm, state0, frc, fns if cfg.forcing_next else None,
                        doy, horizon)
    total, parts = 0.0, []
    for k, p in enumerate(preds):
        lk, pk = loss_fn(p, y[:, k], w)
        total = total + lw[k] * lk
        parts.append(pk)
    return total, parts, preds, w, y, state0


def run_epoch(model, cfg, loader, loss_fn, device, depth_enc, horizon, opt=None,
              scaler=None, amp=False, sched=None, step0=0, smoke=0, log_every=0,
              n_steps=None, tag=""):
    train = opt is not None
    model.train(train)
    lw = lead_weights(cfg, horizon)
    n, step, tot = 0, 0, 0.0
    sums = {}
    t0 = t_prev = time.time()
    wait = 0.0
    for i, batch in enumerate(loader):
        wait += time.time() - t_prev
        batch = to_dev(batch, device)
        if train and sched is not None:
            for g in opt.param_groups:
                g["lr"] = sched(step0 + step)
        with torch.set_grad_enabled(train):
            loss, parts, preds, w, y, state0 = run_batch(
                model, cfg, batch, depth_enc, loss_fn, lw, amp, device, horizon)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
        else:
            with torch.no_grad():
                for k in range(horizon):
                    for who, pk in (("model", parts[k]),
                                    ("pers", loss_fn(state0, y[:, k], w)[1])):
                        for key, v in pk.items():
                            kk = f"{who}_l{k + 1}_{key}"
                            sums[kk] = sums.get(kk, 0.0) + v
        bs = batch[0].shape[0]
        n += bs
        tot += float(loss.detach()) * bs
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            eta = f" eta {(n_steps - i - 1) * el / (i + 1) / 60:.0f} min" if n_steps else ""
            print(f"  [{tag}] step {i + 1}/{n_steps or '?'} loss={tot / n:.4f} "
                  f"{el / (i + 1):.3f}s/step wait={100 * wait / el:.0f}%{eta}", flush=True)
        if smoke and i + 1 >= smoke:
            break
        t_prev = time.time()
    return tot / max(n, 1), sums, step


def skill(sums, tvars, horizon):
    """{(lead, var): (mse, skill)} from pooled val sums."""
    out = {}
    for k in range(1, horizon + 1):
        for v in tvars:
            wsum = sums.get(f"model_l{k}_wsum_{v}", 0.0)
            sm = sums.get(f"model_l{k}_sse_{v}", float("nan"))
            sp = sums.get(f"pers_l{k}_sse_{v}", float("nan"))
            out[(k, v)] = (sm / wsum if wsum > 0 else float("nan"),
                           sm / sp if wsum > 0 and sp > 0 else float("nan"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, nargs="?", const=2, default=0,
                    help="run N batches only (bare --smoke = 2)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = Cfg.load(args.config)                  # asserts C1-C4
    tr = cfg.train
    if tr.get("require_cuda", True) and not torch.cuda.is_available():
        sys.exit("FATAL: no CUDA device. Every ASPIRE 2A GPU queue defaults to 0 GPUs: "
                 "put ngpus=1 in the select= line (e.g. select=1:ngpus=1:ncpus=16:mem=110gb). "
                 "Set train.require_cuda: false to run on CPU deliberately.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.paths["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    H = cfg.horizon
    nw = int(tr["num_workers"])
    print(f"[cfg] {args.config}: grid {cfg.h}x{cfg.w}, in_ch={cfg.in_ch} out_ch={cfg.out_ch}, "
          f"horizon {H}, forcing_next {cfg.forcing_next}, mapped stacks "
          f"<= {projected_ram_gb(cfg, nw):.0f} GB", flush=True)

    frac = 1.0 if args.smoke else float(tr.get("days_per_epoch_frac", 1.0))
    train_ds = WindowStream(args.config, "train", frac=frac, block=int(tr.get("day_block", 8)))
    vfrac = 1.0 if args.smoke else float(tr.get("val_days_frac", 0.1))
    val_ds = WindowStream(args.config, "val", frac=vfrac, block=1)

    def loader(ds):
        return DataLoader(ds, batch_size=int(tr["batch_size"]), num_workers=nw,
                          pin_memory=device.type == "cuda", drop_last=True,
                          persistent_workers=nw > 0)

    train_dl, val_dl = loader(train_ds), loader(val_ds)
    model = build_model_from_cfg(cfg).to(device)
    loss_fn = StateLoss(cfg, (cfg.raw.get("loss") or {}).get("var_weights")).to(device)
    depth_enc = torch.from_numpy(depth_encoding(cfg.depth_m, cfg.patch, cfg.patch)).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)

    init = tr.get("init_from") or ""
    ckpt = os.path.join(out_dir, "last.pt")
    resuming = os.path.exists(ckpt) and not args.smoke and not args.no_resume
    if init and not resuming:
        if not os.path.exists(init):
            sys.exit(f"train.init_from {init} does not exist: train the one-day model first")
        ck = torch.load(init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[init] weights from {init} (epoch {ck.get('epoch')})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]),
                            weight_decay=float(tr["weight_decay"]))
    amp = bool(tr.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epochs = 1 if args.smoke else int(tr["epochs"])
    steps_ep = max(1, len(train_ds) // int(tr["batch_size"]))
    total = steps_ep * epochs
    warm = int(tr.get("warmup_steps", 500))

    def sched(s):
        return lr_at(s, warm, float(tr["lr"]), total)

    start, best, gstep = 0, float("inf"), 0
    if resuming:
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best, gstep = ck["epoch"] + 1, ck.get("best_val", best), ck.get("gstep", 0)
        print(f"[resume] epoch {start}, step {gstep}", flush=True)
    train_ds.epoch = start

    tvars = loss_fn.target_vars
    header = (["epoch", "train_loss", "val_loss", "lr", "sec"]
              + [f"val_skill_l{k}_{v}" for k in range(1, H + 1) for v in tvars]
              + [f"val_mse_l{k}_{v}" for k in range(1, H + 1) for v in tvars])
    log = os.path.join(out_dir, "metrics.csv")
    if not args.smoke and not os.path.exists(log):
        with open(log, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def save(path, ep):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "best_val": best, "gstep": gstep, "cfg": cfg.raw,
                    "config_path": os.path.abspath(args.config)}, path)

    for ep in range(start, epochs):
        t0 = time.time()
        trl, _, ns = run_epoch(model, cfg, train_dl, loss_fn, device, depth_enc, H, opt,
                               scaler, amp, sched, gstep, args.smoke,
                               int(tr.get("log_every", 200)), steps_ep, f"ep {ep:03d}")
        gstep += ns
        val, sums, _ = run_epoch(model, cfg, val_dl, loss_fn, device, depth_enc, H,
                                 amp=amp, smoke=args.smoke)
        sk = skill(sums, tvars, H)
        dt = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        print(f"[ep {ep:03d}] train={trl:.4f} val={val:.4f} lr={lr:.2e} ({dt:.0f}s)")
        for k in range(1, H + 1):
            print(f"         day {k} skill vs persistence (<1 beats it): "
                  + "  ".join(f"{v}={sk[(k, v)][1]:.3f}" for v in tvars), flush=True)
        if args.smoke:
            continue
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep, trl, val, lr, f"{dt:.0f}"]
                                   + [sk[(k, v)][1] for k in range(1, H + 1) for v in tvars]
                                   + [sk[(k, v)][0] for k in range(1, H + 1) for v in tvars])
        if val < best:
            best = val
            save(os.path.join(out_dir, "best.pt"), ep)
            print(f"  -> new best {best:.4f}", flush=True)
        save(ckpt + ".part", ep)
        os.replace(ckpt + ".part", ckpt)
    print("done." + (f" (smoke={args.smoke})" if args.smoke else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
