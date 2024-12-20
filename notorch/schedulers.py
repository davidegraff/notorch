from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from notorch.types import LRSchedConfig


def meta_lr_sched_factory(
    scheduler: Callable[[Optimizer], LRScheduler], config: LRSchedConfig
) -> Callable[[Optimizer], LRSchedConfig]:
    def fun(optim: Optimizer):
        return config | {"scheduler": scheduler(optim)}

    return fun


def NoamLikeLRSched(
    optimizer: Optimizer,
    warmup_steps: int,
    cooldown_steps: int,
    init_lr: float,
    max_lr: float,
    final_lr: float,
) -> LambdaLR:
    r"""A learning rate scheduler that is _like_ Noam scheduler from [1]_ (sec. 5.3).

    The learning rate is scheduled via piecewise linear warmup followed by exponential decay.
    Specficially, it increases linearly from ``init_lr`` to ``max_lr`` over :attr:`warmup_steps`
    steps then decreases exponentially to ``final_lr`` over ``total_steps - warmup_steps`` (where
    ``total_steps = total_epochs * steps_per_epoch``).

    Formally, the learning rate is defined as:

    .. math::
        \mathtt{lr}(i) &=
            \begin{cases}
                \mathtt{init\_lr} + \delta \cdot i &\text{if } i < \mathtt{warmup\_steps} \\
                \mathtt{max\_lr} \cdot \left( \frac{\mathtt{final\_lr}}{\mathtt{max\_lr}} \right)^{
                    \gamma(i)} &\text{otherwise
                } \\
            \end{cases}
        \\
        \delta &\equiv
            \frac{\mathtt{max\_lr} - \mathtt{init\_lr}}{\mathtt{warmup\_steps}} \\
        \gamma(i) &\coloneqq
            \frac{i - \mathtt{warmup\_steps}}{\mathtt{total\_steps} - \mathtt{warmup\_steps}}

    Parameters
    -----------
    optimizer : Optimizer
        A PyTorch optimizer.
    warmup_steps : int
        The number of steps during which to linearly increase the learning rate.
    cooldown_steps : int
        The number of steps during which to exponential decay the learning rate.
    init_lr : float
        The initial learning rate.
    max_lr : float
        The maximum learning rate (achieved after ``warmup_steps``).
    final_lr : float
        The final learning rate (achieved after ``cooldown_steps``).

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł.
    and Polosukhin, I. "Attention is all you need." Advances in neural information processing
    systems, 2017, 30. https://arxiv.org/abs/1706.03762

    Notes
    -----
    - this is directly copied from
    https://github.com/chemprop/chemprop/blob/fa2d6dceb054ced6fe46191a628cfd664037d5e7/chemprop/schedulers.py
    """

    def lr_lambda(step: int):
        if step < warmup_steps:
            warmup_factor = (max_lr - init_lr) / warmup_steps
            return step * warmup_factor / init_lr + 1
        elif warmup_steps <= step < warmup_steps + cooldown_steps:
            cooldown_factor = (final_lr / max_lr) ** (1 / cooldown_steps)
            return (max_lr * (cooldown_factor ** (step - warmup_steps))) / init_lr
        else:
            return final_lr / init_lr

    return LambdaLR(optimizer, lr_lambda)
