from abc import ABCMeta, abstractmethod


class BaseLR:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter):
        pass


class PolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_iter):
        return self.start_lr * (
            (1 - float(cur_iter) / self.total_iters) ** self.lr_power
        )


class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                (1 - float(cur_iter) / self.total_iters) ** self.lr_power
            )


class MultiStageLR(BaseLR):
    def __init__(self, lr_stages):
        assert (
            type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2
        ), "lr_stages must be list or tuple, with [iters, lr] format"
        self._lr_stagess = lr_stages

    def get_lr(self, epoch):
        for it_lr in self._lr_stagess:
            if epoch < it_lr[0]:
                return it_lr[1]


class CosineWarmRestartsLR(BaseLR):
    """Cosine annealing with warm restarts (SGDR, Loshchilov & Hutter 2017).

    Args:
        start_lr: Peak learning rate at each restart.
        min_lr: Minimum learning rate at the end of each cosine cycle.
        total_iters: Total number of training iterations.
        cycle_iters: Iterations per cosine cycle (T_0).
        cycle_mult: Multiplier for cycle length after each restart (T_mult).
            1 = fixed-length cycles, 2 = doubling cycles, etc.
        warmup_steps: Linear warmup iterations at the very start of training.
    """

    def __init__(self, start_lr, min_lr, total_iters, cycle_iters,
                 cycle_mult=1, warmup_steps=0):
        import math
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.total_iters = total_iters
        self.cycle_iters = cycle_iters
        self.cycle_mult = cycle_mult
        self.warmup_steps = warmup_steps
        self._math = math

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)

        t = cur_iter - self.warmup_steps
        if self.cycle_mult == 1:
            cycle_pos = t % self.cycle_iters
            cycle_len = self.cycle_iters
        else:
            # Geometric series: find which cycle we're in
            cycle = 0
            consumed = 0
            cycle_len = self.cycle_iters
            while consumed + cycle_len <= t:
                consumed += cycle_len
                cycle += 1
                cycle_len = int(self.cycle_iters * (self.cycle_mult ** cycle))
            cycle_pos = t - consumed

        cos_val = self._math.cos(self._math.pi * cycle_pos / cycle_len)
        return self.min_lr + 0.5 * (self.start_lr - self.min_lr) * (1 + cos_val)


class LinearIncreaseLR(BaseLR):
    def __init__(self, start_lr, end_lr, warm_iters):
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._warm_iters = warm_iters
        self._delta_lr = (end_lr - start_lr) / warm_iters

    def get_lr(self, cur_epoch):
        return self._start_lr + cur_epoch * self._delta_lr
