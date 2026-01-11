import math
from collections.abc import Callable, Iterable

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.optim.optimizer import ParamsT

from cs336_basics.model import Embedding


def cross_entropy(
    inputs: Float[Tensor, "batch ... vocab_size"],
    targets: Int[Tensor, "batch ..."]
) -> Float[Tensor, ""]:
    """
    compute the cross entropy loss, which takes in predicted logits (oi) and targets (xi+1)
    and computes the cross entropy.

    Derivation:
    CE = -log(exp(x_target) / Σ exp(x_j))
       = log(Σ exp(x_j)) - x_target
       = logsumexp(x) - x_target
    """
    logits = rearrange(inputs, "b ... v -> (b ...) v")
    indices = rearrange(targets, "b ... -> (b ...)")

    # don't need logits = logits - torch.amax(logits, dim=-1, keepdim=True)
    # logsumexp() has already subtract the largest element for numerical stability
    logsumexp = torch.logsumexp(logits, dim = -1)

    # use both batch and class indices to index the correct logits
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    gt_logits = logits[batch_idx, indices]
    return torch.mean(logsumexp - gt_logits)


def perplexity(
    inputs: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    return torch.exp(cross_entropy(inputs, targets))


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]          # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data    # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1    # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    """
        ADAMW: DECOUPLED ADAPTIVE MOMENT ESTIMATION

        1. MOMENTUM (1st Moment - Inertia):
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        Directs the update using smoothed historical gradients.

        2. RMSprop (2nd Moment - Scale):
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        Normalizes step size: dampens large gradients, boosts small ones.

        3. BIAS CORRECTION (Cold Start):
        m_hat = m_t / (1 - β1^t)
        v_hat = v_t / (1 - β2^t)
        Compensates for the 0-initialization bias in early steps.

        4. WEIGHT DECAY (Decoupled):
        θ = θ - (η * λ * θ)
        Applied directly to weights, NOT via gradients, to prevent 
        v_t from suppressing the regularization effect.

        5. FINAL UPDATE (Scalar Folding):
        Combined Step Size: η_eff = η * sqrt(1 - β2^t) / (1 - β1^t)
        Update: θ_{t+1} = (θ_t - η * λ * θ_t) - η_eff * m_t / (sqrt(v_t) + ε)
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 1e-2
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """
        in place update, leverage step to group param, use torch._foreach_* speedup
        """
        # callable closure to re-compute the loss before the optimizer step, not used
        loss = None if closure is None else closure()
        # params group is params manager for tuning param
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            decay = group["decay"]

            # leverage step to group param, and get to calculate lr_t
            if 'step' not in group:
                group["step"] = 0
            group['step'] += 1
            t = group['step']

            # avoid in for each param, is cheap for each param_group
            # lr_t = lr * (sqrt(1-b2^t) / (1-b1^t))
            lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
            # weight dency: θ = θ - lr * decay * θ
            dency_factor = 1 - lr * decay

            # repeat for each param
            # should optimize with _foreach_* to aviod many small kernel launch
            for p in group["params"]:
                p: nn.Parameter
                if p.grad is None:
                    continue

                # for update in place, save to state first, then get from state
                # the state is like a tarin daily to record momentum/variance etc.
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                grad: Tensor = p.grad
                m: Tensor = state["m"]
                v: Tensor = state["v"]

                # m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = beta2 * v + (1 - beta2) * grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # weight dency: θ = θ - lr * decay * θ
                p.mul_(dency_factor)
                # θ = θ - lr_t * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-lr_t)

        return loss



def get_lr_cosine_schedule(
    it: int,
    lr_max: float,
    lr_min: float,
    t_w: int, # warmup_iters
    t_c: int, # cosine_cycle_iters
) -> float:
    """
        Given the parameters of a cosine learning rate decay schedule (with linear
        warmup) and an iteration number, return the learning rate at the given
        iteration under the specified schedule.

        Args:
            it (int): Iteration number to get learning rate for.
            max_learning_rate (float): alpha_max, the maximum learning rate for
                cosine learning rate schedule (with warmup).
            min_learning_rate (float): alpha_min, the minimum / final learning rate for
                the cosine learning rate schedule (with warmup).
            warmup_iters (int): T_w, the number of iterations to linearly warm-up
                the learning rate.
            cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

        Returns:
            Learning rate at the given iteration under the specified schedule.
    """
    # Warm-up
    if it < t_w:
        return it / t_w * lr_max
    # Cosine annealing
    elif it <= t_c:
        cos_anneal = 1 + math.cos((it - t_w) / (t_c - t_w) * math.pi)
        return lr_min + 0.5 * cos_anneal * (lr_max - lr_min)
    # Post annealing
    else:
        return lr_min


def get_lr_wsd_schedule(
    it: int,
    lr_max: float,
    lr_min: float,
    steps: int,  # total steps
    t_w: int,  # warm up step
    t_c: int,  # cosine decay step
    decay_ratio: float = 0.1  # experience golden ratio
) -> float:
    """
    WSD (Warmup-Stable-Decay) Scheduler
    """
    # Warmup <5%
    if it < t_w:
        return lr_max * (it + 1) / (t_w + 1)

    decay_steps = steps - t_c

    # Stable (High Water Level) ~90%
    if it < t_c:
        return lr_max

    # Decay (Rapid Drop) ~10%
    progress = (it - t_c) / decay_steps
    progress = min(1.0, max(0.0, progress))
    # cosine decay
    cos_decay = 1.0 + math.cos(progress * math.pi)
    return lr_min + 0.5 * cos_decay * (lr_max - lr_min)


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> torch.Tensor:
    # get all parameters
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)

    # Compute the L2 norm for each gradient tensor efficiently using vectorized operations
    norms = torch._foreach_norm(grads, 2.0)
    # Aggregate individual norms to compute the global L2 norm of the entire gradient vector
    total_l2_norm = torch.linalg.vector_norm(torch.stack(norms), 2.0)

    if total_l2_norm > max_l2_norm:
        scale = max_l2_norm / (total_l2_norm + eps)
        torch._foreach_mul_(grads, scale)

    return total_l2_norm


# muon code from: https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G: Tensor, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G if G.dtype == torch.bfloat16 else G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def muon_update(grad: Tensor, momentum: Tensor, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def adam_update(grad: Tensor, buf1: Tensor, buf2: Tensor, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer with automatic parameter grouping.
    
    Automatically splits parameters into:
    - Muon group: 2D+ weight matrices from Linear/Conv layers
    - AdamW group: 1D parameters (norms, biases) and Embedding weights
    
    Usage:
        optimizer = Muon(model, weight_decay=0.01, adam_lr=3e-4)
    """

    def __init__(
        self,
        model: nn.Module,
        weight_decay: float = 0.01,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        **kwrags,
    ):
        # use lr_ratio to maintain the relative scale between Muon and Adam learning rates
        # force ensures consistent Muon and Adam learning rates updates
        self.lr_ratio = muon_lr / lr
        muon_params = []
        adam_params = []
        processed_ids = set()

        # record the module that don't use muon
        muon_skip_ids = set()
        muon_skip_class = (Embedding, )
        for _, m in model.named_modules():
            if isinstance(m, muon_skip_class):
                for p in m.parameters():
                    muon_skip_ids.add(id(p))

        # process all parameters
        for _, p in model.named_parameters():
            # skip frozened params
            if not p.requires_grad:
                continue

            # for tie weights
            param_id = id(p)
            if param_id in processed_ids:
                continue
            processed_ids.add(param_id)

            # add to suitable params list, muon works with 2d params
            if p.ndim >= 2 and id(p) not in muon_skip_ids:
                muon_params.append(p)
            else:
                adam_params.append(p)

        # setup param_groups and defaults
        param_groups = []
        if muon_params:
            param_groups.append({
                "params": muon_params,
                "use_muon": True,
                "lr": lr,  # use lr_ratio to scale to muon_lr
                "momentum": muon_momentum,
                "weight_decay": weight_decay,
            })
        if adam_params:
            param_groups.append({
                "params": adam_params,
                "use_muon": False,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            })

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(param_groups, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            adam_lr = group["lr"]
            if group["use_muon"]:
                for p in group["params"]:
                    p: nn.Parameter
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - adam_lr * self.lr_ratio * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-adam_lr * self.lr_ratio)

            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1

                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - adam_lr * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
