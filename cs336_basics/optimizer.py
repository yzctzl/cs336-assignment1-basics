import math
from collections.abc import Callable, Iterable

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.optim.optimizer import ParamsT


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
    max_lr: float,
    min_lr: float,
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
        return it / t_w * max_lr
    # Cosine annealing
    elif it <= t_c:
        cos_anneal = 1 + math.cos((it - t_w) / (t_c - t_w) * math.pi)
        return min_lr + 0.5 * cos_anneal * (max_lr - min_lr)
    # Post annealing
    else:
        return min_lr


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
