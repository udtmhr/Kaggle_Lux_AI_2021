from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearSchedule:
    start: float
    end: float
    duration: int
    delay: int = 0

    def __call__(self, step: int) -> float:
        if self.duration <= 0:
            return self.end
        progress = min(max((int(step) - self.delay) / self.duration, 0.0), 1.0)
        return self.start + progress * (self.end - self.start)


def teacher_kl_coefficient(flags, step: int) -> float:
    if not getattr(flags, "use_teacher", False):
        return 0.0
    start = float(getattr(flags, "teacher_kl_cost_start", flags.teacher_kl_cost))
    end = float(getattr(flags, "teacher_kl_cost_end", flags.teacher_kl_cost))
    duration = int(float(getattr(flags, "teacher_kl_decay_steps", 0)))
    delay = int(float(getattr(flags, "teacher_kl_delay_steps", 0)))
    floor = float(getattr(flags, "teacher_kl_cost_floor", 0.0))
    if floor < 0.0:
        raise ValueError("teacher_kl_cost_floor must be non-negative")
    return max(LinearSchedule(start, end, duration, delay)(step), floor)
