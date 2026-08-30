"""
hhpc_base.py
=============
Base HH-PC model extracted from the ablation study.

This is the core two-layer Hodgkin-Huxley spiking network trained with
Predictive Coding (PC): input -> hidden (spiking) -> output (spiking).
Both hidden and output layers are HH neurons.

Removed relative to the ablation source (these were ablation-only):
    - LIF neuron variant, analog (non-spiking) variant
    - N-MNIST / Caltech loaders, KMNIST mirror-patching fallbacks
    - ablation suites (ablate_standard / ablate_nmnist / ablate_caltech),
      CSV table writers, ROC-AUC computation, CFG/run_ablations driver

Kept as-is: HHNeuron dynamics, PC inference/learning (neuron-agnostic),
metrics, spike-rate accounting, and the training loop.
"""

import os
import time
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ─────────────────────────────────────────────────────────────────────────
# 0. UTILITIES
# ─────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y.long(), num_classes=num_classes).float()


# ─────────────────────────────────────────────────────────────────────────
# 1. DATA LOADER (MNIST / FMNIST / KMNIST via torchvision)
# ─────────────────────────────────────────────────────────────────────────

def get_loaders(
    dataset_name: str = "MNIST",
    batch_size: int = 128,
    root: str = "./data",
    device: str = "cpu",
    val_ratio: float = 0.1,
    seed: int = 42,
):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = dataset_name.upper()

    if ds == "KMNIST":
        train_full = datasets.KMNIST(root=root, train=True, download=True, transform=tfm)
        test_ds = datasets.KMNIST(root=root, train=False, download=True, transform=tfm)
    elif ds in ("FMNIST", "FASHIONMNIST"):
        train_full = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
        test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=tfm)
    else:
        train_full = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    n_total = len(train_full)
    n_val = max(1, int(round(val_ratio * n_total)))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=g)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    kw = dict(num_workers=2 if use_cuda else 0, pin_memory=use_cuda)

    return (
        DataLoader(train_ds, batch_size, shuffle=True, **kw),
        DataLoader(val_ds, batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size, shuffle=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. HODGKIN-HUXLEY NEURON
# ─────────────────────────────────────────────────────────────────────────

class HHNeuron(nn.Module):
    class Gate:
        def __init__(self, B, N, device):
            self.alpha = torch.zeros(B, N, device=device)
            self.beta = torch.zeros(B, N, device=device)
            self.state = torch.zeros(B, N, device=device)

        def update(self, dt):
            s = self.state + dt * (self.alpha * (1 - self.state) - self.beta * self.state)
            return s.clamp(0.0, 1.0)

        def set_inf(self):
            self.state = self.alpha / (self.alpha + self.beta + 1e-8)

    def __init__(self, N, dt=0.03, device="cpu", thr=0.8, reset=0.0, tau_ref=2.0):
        super().__init__()
        self.N, self.dt, self.device = N, float(dt), device
        self.thr, self.reset = float(thr), float(reset)
        self.refr_steps = max(1, int(round(tau_ref / dt)))
        for name, val in [("ENa", 115.), ("EK", -12.), ("Eleak", 10.6),
                           ("gNa", 120.), ("gK", 36.), ("gLeak", 0.3), ("Cm", 1.)]:
            self.register_buffer(name, torch.tensor(val))
        self.reset_states(1)

    def reset_states(self, B):
        dev = self.device
        self.B = B
        self.Vm = torch.zeros(B, self.N, device=dev)
        self.m = HHNeuron.Gate(B, self.N, dev)
        self.n = HHNeuron.Gate(B, self.N, dev)
        self.h = HHNeuron.Gate(B, self.N, dev)
        self._update_gates(self.Vm)
        self.m.set_inf(); self.n.set_inf(); self.h.set_inf()
        self.refr = torch.zeros(B, self.N, device=dev)

    def _update_gates(self, V):
        V = V.clamp(-100., 100.)
        self.n.alpha = 0.01 * (10 - V) / (torch.exp((10 - V) / 10) - 1 + 1e-8)
        self.n.beta = 0.125 * torch.exp(-V / 80.)
        self.m.alpha = 0.1 * (25 - V) / (torch.exp((25 - V) / 10) - 1 + 1e-8)
        self.m.beta = 4. * torch.exp(-V / 18.)
        self.h.alpha = 0.07 * torch.exp(-V / 20.)
        self.h.beta = 1. / (torch.exp((30 - V) / 10) + 1.)

    def forward(self, I):
        if I.shape[0] != self.B:
            self.reset_states(I.shape[0])
        self._update_gates(self.Vm)
        m = self.m.update(self.dt); n = self.n.update(self.dt); h = self.h.update(self.dt)
        INa = m ** 3 * self.gNa * h * (self.Vm - self.ENa)
        IK = n ** 4 * self.gK * (self.Vm - self.EK)
        IL = self.gLeak * (self.Vm - self.Eleak)
        dV = (I - INa - IK - IL) / self.Cm
        Vn = self.Vm + self.dt * dV
        Vn = torch.tanh(Vn / 30.) * 30.
        can = (self.refr <= 0)
        spk = ((Vn >= self.thr) & can).float()
        self.Vm = torch.where(spk.bool(), torch.full_like(Vn, self.reset), Vn)
        self.m.state = m; self.n.state = n; self.h.state = h
        self.refr = torch.where(spk.bool(),
                                 torch.full_like(self.refr, float(self.refr_steps)),
                                 (self.refr - 1.).clamp(min=0.))
        return spk, self.Vm


# ─────────────────────────────────────────────────────────────────────────
# 3. PC ACTIVATION
# ─────────────────────────────────────────────────────────────────────────

def make_pc_activation(name: str):
    name = name.lower()
    if name == "sigmoid":
        return torch.sigmoid, lambda z: torch.sigmoid(z) * (1 - torch.sigmoid(z))
    if name == "tanh":
        return torch.tanh, lambda z: 1 - torch.tanh(z) ** 2
    def f(z): return z.clamp(0., 1.)
    def fp(z): return ((z > 0.) & (z < 1.)).float()
    return f, fp


# ─────────────────────────────────────────────────────────────────────────
# 4. HH-PC NETWORK (two spiking layers: hidden + output)
# ─────────────────────────────────────────────────────────────────────────

class PCSNNet(nn.Module):
    """Two-layer HH-PC network: input -> hidden (HH, spiking) -> output
    (HH, spiking). input_encoding = "poisson" | "latency_first"."""

    def __init__(
        self,
        layer_sizes: List[int],
        dt: float = 0.03,
        device: str = "cpu",
        current_gain: float = 30.0,
        I_bias: float = 2.0,
        thr: float = 0.8,
        pc_activation: str = "relu",
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        input_encoding: str = "poisson",
        poisson_scale: float = 1.0,
    ):
        super().__init__()
        assert len(layer_sizes) >= 2
        self.device = torch.device(device)
        self.sizes = layer_sizes
        self.dt = float(dt)
        self.current_gain = float(current_gain)
        self.I_bias = float(I_bias)
        self.input_encoding = input_encoding.lower()
        self.poisson_scale = float(poisson_scale)
        self.f, self.fprime = make_pc_activation(pc_activation)
        self.L = len(layer_sizes) - 1
        self.S = self.L  # both layers spike

        self.syn = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True)
            for i in range(self.L)
        ])
        for lin in self.syn:
            nn.init.xavier_uniform_(lin.weight, gain=0.5)
            nn.init.zeros_(lin.bias)

        self.cells = nn.ModuleList([
            HHNeuron(layer_sizes[i + 1], dt=dt, device=device, thr=thr)
            for i in range(self.S)
        ])

        self.opt = torch.optim.Adam(self.syn.parameters(), lr=lr, weight_decay=weight_decay)
        self._last_spike_sums: Optional[List[torch.Tensor]] = None

    @torch.no_grad()
    def _encode_latency(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        lat = steps * (1.0 - x0.clamp(0, 1))
        return lat.clamp(0., float(steps))

    @torch.no_grad()
    def _build_spike_train(self, latencies: torch.Tensor, steps: int) -> torch.Tensor:
        tgrid = torch.arange(1, steps + 1, device=self.device).view(1, 1, -1)
        return (latencies.unsqueeze(-1) <= tgrid).float()

    @torch.no_grad()
    def forward_proxies(self, x_in: torch.Tensor, steps_spk: int) -> List[torch.Tensor]:
        x0 = x_in.to(self.device).clamp(0, 1)
        B = x0.size(0)

        for cell in self.cells:
            cell.reset_states(B)

        if self.input_encoding == "latency_first":
            lat = self._encode_latency(x0, steps_spk)
            spk_train = self._build_spike_train(lat, steps_spk)
            fired = torch.zeros_like(x0, dtype=torch.bool)
        else:
            p = (x0 * self.poisson_scale).clamp(0, 1)
            spk_train = (torch.rand(B, x0.size(1), steps_spk,
                                     device=self.device) < p.unsqueeze(-1)).float()

        spike_sums = [torch.zeros(B, self.sizes[i + 1], device=self.device)
                      for i in range(self.S)]

        for t in range(steps_spk):
            if self.input_encoding == "poisson":
                inp = spk_train[:, :, t]
            else:
                inp = (spk_train[:, :, t] * (~fired)).float()
                fired.logical_or_(inp.bool())

            r = inp
            for i in range(self.S):
                h = F.linear(r, self.syn[i].weight, self.syn[i].bias)
                I = h * self.current_gain + self.I_bias
                spk, _ = self.cells[i](I)
                spike_sums[i] += spk
                r = spk

        self._last_spike_sums = spike_sums
        proxies = [x0] + [(ss / float(steps_spk)).clamp(0, 1) for ss in spike_sums]
        return proxies

    def last_spike_sums(self):
        return self._last_spike_sums

    # ── PC inference / learning (neuron-agnostic) ──────────────────────

    def pc_infer(self, x_init, y_target=None, T_infer=50, eta_x=0.05, clamp_output=True):
        L = self.L
        x = [xi.clone().detach().to(self.device) for xi in x_init]
        x[0] = x[0].clamp(0, 1)
        if clamp_output and y_target is not None:
            x[L] = y_target.clone().detach().to(self.device).clamp(0, 1)

        z_cache = [None] * L
        for _ in range(T_infer):
            e = [None] * (L + 1)
            e[0] = torch.zeros_like(x[0])
            for l in range(1, L + 1):
                idx = l - 1
                z = F.linear(x[l - 1], self.syn[idx].weight, self.syn[idx].bias)
                z_cache[idx] = z
                e[l] = x[l] - self.f(z)
            for l in range(1, L):
                fb = (e[l + 1] * self.fprime(z_cache[l])) @ self.syn[l].weight
                x[l] = (x[l] - eta_x * (e[l] - fb)).clamp_(0, 1)
            if not clamp_output:
                x[L] = (x[L] - eta_x * e[L]).clamp_(0, 1)

        energy = 0.
        with torch.no_grad():
            for l in range(1, L + 1):
                idx = l - 1
                z = F.linear(x[l - 1], self.syn[idx].weight, self.syn[idx].bias)
                el = x[l] - self.f(z)
                energy += 0.5 * (el ** 2).mean().item()
        return x, e, z_cache, energy

    def pc_learn(self, x, e, z_cache):
        B = x[0].shape[0]
        self.opt.zero_grad()
        for idx in range(self.L):
            local = e[idx + 1] * self.fprime(z_cache[idx])
            self.syn[idx].weight.grad = -(local.T @ x[idx]) / B
            self.syn[idx].bias.grad = -local.mean(0)
        torch.nn.utils.clip_grad_norm_(self.syn.parameters(), 1.0)
        self.opt.step()

    def train_step(self, x_in, y_target, steps_spk, T_infer, eta_x):
        proxies = self.forward_proxies(x_in, steps_spk)
        x, e, z_cache, energy = self.pc_infer(proxies, y_target, T_infer, eta_x, True)
        self.pc_learn(x, e, z_cache)
        return energy, proxies


# ─────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    def __init__(self, C, device="cpu"):
        self.C = C; self.device = device; self.reset()

    def reset(self):
        self.tp = torch.zeros(self.C, dtype=torch.long)
        self.fp = torch.zeros(self.C, dtype=torch.long)
        self.fn = torch.zeros(self.C, dtype=torch.long)
        self.correct = self.total = 0

    @torch.no_grad()
    def update(self, pred, y):
        pred = pred.view(-1).long(); y = y.view(-1).long()
        self.total += y.numel()
        self.correct += int((pred == y).sum())
        tp = torch.bincount(pred[pred == y], minlength=self.C)
        pc = torch.bincount(pred, minlength=self.C)
        tc = torch.bincount(y, minlength=self.C)
        self.tp += tp; self.fp += pc - tp; self.fn += tc - tp

    def compute(self, eps=1e-8):
        tp = self.tp.float(); fp = self.fp.float(); fn = self.fn.float()
        P = (tp / (tp + fp + eps)).mean().item()
        R = (tp / (tp + fn + eps)).mean().item()
        F1 = (2 * tp / (2 * tp + fp + fn + eps)).mean().item()
        return {"acc": self.correct / max(self.total, 1), "precision": P, "recall": R, "f1": F1}


# ─────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def spike_rate_epoch(model, loader, device, steps_spk, eval_seed=1234):
    model.eval()
    S = model.S
    total_spk = [0.] * S
    total_den = [0.] * S
    n_samples = 0

    with torch.random.fork_rng():
        torch.manual_seed(eval_seed)
        for x, _ in loader:
            x = x.to(device).view(x.size(0), -1)
            B = x.size(0)
            model.forward_proxies(x, steps_spk)
            ss = model.last_spike_sums()
            if ss is None:
                continue
            for li in range(S):
                total_spk[li] += float(ss[li].sum())
                total_den[li] += float(B * ss[li].shape[1] * steps_spk)
            n_samples += B

    per_layer = [total_spk[li] / max(total_den[li], 1.) for li in range(S)]
    total = sum(total_spk) / max(sum(total_den), 1.)
    sps = sum(total_spk) / max(n_samples, 1)
    return {"per_layer": per_layer, "total": total, "spikes_per_sample": sps}


@torch.no_grad()
def eval_epoch(model, loader, device, steps_spk, T_infer, eta_x, eval_mode="pc", eval_seed=1234):
    model.eval()
    C = model.sizes[-1]
    ff_acc = MetricAccumulator(C)
    pc_acc = MetricAccumulator(C)
    total_e, total = 0., 0

    with torch.random.fork_rng():
        torch.manual_seed(eval_seed)
        for x, y in loader:
            x = x.to(device).view(x.size(0), -1)
            y = y.to(device)
            B = x.size(0)
            proxies = model.forward_proxies(x, steps_spk)

            ff_pred = proxies[-1].argmax(1)
            ff_acc.update(ff_pred.cpu(), y.cpu())

            if eval_mode == "pc":
                xs, _, _, _ = model.pc_infer(proxies, None, T_infer, eta_x, False)
                pc_pred = xs[-1].argmax(1)
            else:
                pc_pred = ff_pred
            pc_acc.update(pc_pred.cpu(), y.cpu())

            y_oh = one_hot(y, C)
            _, _, _, e = model.pc_infer(proxies, y_oh, T_infer, eta_x, True)
            total_e += e * B; total += B

    ff_m = ff_acc.compute(); pc_m = pc_acc.compute()
    return {
        "ff_acc": ff_m["acc"], "ff_f1": ff_m["f1"],
        "pc_acc": pc_m["acc"], "pc_f1": pc_m["f1"],
        "pc_precision": pc_m["precision"], "pc_recall": pc_m["recall"],
        "pc_energy": total_e / max(total, 1),
    }


# ─────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best = None; self.bad = 0; self.best_epoch = 0

    def step(self, val, epoch):
        if self.best is None or val > self.best + self.min_delta:
            self.best = val; self.bad = 0; self.best_epoch = epoch
            return False, True
        self.bad += 1
        return (self.bad >= self.patience), False


def train(model, train_loader, val_loader, device,
          steps_spk, T_infer_train, T_infer_eval,
          eta_x, epochs, eval_mode, eval_seed,
          patience, ckpt_path, verbose=False):

    stopper = EarlyStopper(patience=patience)
    best_val_acc = 0.

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device).view(x.size(0), -1)
            y = y.to(device)
            model.train_step(x, one_hot(y, model.sizes[-1]), steps_spk, T_infer_train, eta_x)

        val_stats = eval_epoch(model, val_loader, device, steps_spk, T_infer_eval, eta_x,
                                eval_mode, eval_seed)
        monitor = val_stats["pc_acc"] if eval_mode == "pc" else val_stats["ff_acc"]

        stop, improved = stopper.step(monitor, epoch)
        if improved:
            torch.save(model.state_dict(), ckpt_path)
            best_val_acc = monitor
        if verbose:
            print(f"  Epoch {epoch:02d} | val_acc={monitor:.4f}{' *' if improved else ''}")
        if stop:
            if verbose:
                print(f"  Early stop at epoch {epoch}")
            break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────

def run_hhpc(
    dataset: str = "MNIST",
    hidden: int = 512,
    epochs: int = 15,
    steps_spk: int = 50,
    T_infer_train: int = 100,
    T_infer_eval: int = 50,
    eta_x: float = 0.05,
    batch_size: int = 128,
    device: Optional[str] = None,
    data_root: str = "./data",
    ckpt_path: str = "hhpc_ckpt.pt",
    verbose: bool = True,
) -> Dict[str, Any]:
    device = device or default_device()
    set_seed(42)

    train_loader, val_loader, test_loader = get_loaders(dataset, batch_size, data_root, device)
    input_dim, num_classes = 28 * 28, 10
    layer_sizes = [input_dim, hidden, num_classes]

    model = PCSNNet(layer_sizes=layer_sizes, device=device).to(device)

    t0 = time.time()
    train(model, train_loader, val_loader, device,
          steps_spk, T_infer_train, T_infer_eval, eta_x,
          epochs, "pc", 1234, 3, ckpt_path, verbose)
    elapsed = time.time() - t0

    test_stats = eval_epoch(model, test_loader, device, steps_spk, T_infer_eval, eta_x, "pc", 1234)
    spike_stats = spike_rate_epoch(model, test_loader, device, steps_spk, 1234)

    print(f"\n[{dataset}] PC acc={test_stats['pc_acc']*100:.2f}%  "
          f"FF acc={test_stats['ff_acc']*100:.2f}%  "
          f"spike_rate={spike_stats['total']:.4f}  train_time={elapsed:.1f}s")

    return {"model": model, "test_stats": test_stats, "spike_stats": spike_stats}


if __name__ == "__main__":
    run_hhpc()
