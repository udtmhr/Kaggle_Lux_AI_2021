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


def rule_prior_alpha(flags, entity: str, step: int) -> float:
    """Return the runtime rule-prior alpha for one action head.

    When no decay fields are configured this preserves the historical fixed
    alpha behavior.  Per-head values take precedence over the global values.
    """
    global_start = float(getattr(flags, "rule_prior_alpha", 0.0))
    start = float(getattr(flags, f"rule_prior_alpha_{entity}", global_start))
    global_end = float(getattr(flags, "rule_prior_alpha_end", global_start))
    end = float(getattr(flags, f"rule_prior_alpha_{entity}_end", global_end))
    duration = int(float(getattr(flags, "rule_prior_decay_steps", 0)))
    delay = int(float(getattr(flags, "rule_prior_decay_delay_steps", 0)))
    if duration <= 0:
        return start
    return LinearSchedule(start, end, duration, delay)(step)


def rule_prior_distill_coefficient(flags, step: int) -> float:
    if not getattr(flags, "rule_prior_distill_enabled", False):
        return 0.0
    start = float(getattr(flags, "rule_prior_distill_cost", 0.0))
    end = float(getattr(flags, "rule_prior_distill_cost_end", start))
    duration = int(float(getattr(flags, "rule_prior_distill_decay_steps", 0)))
    delay = int(float(getattr(flags, "rule_prior_distill_decay_delay_steps", 0)))
    if start < 0.0 or end < 0.0:
        raise ValueError("rule-prior distillation costs must be non-negative")
    return LinearSchedule(start, end, duration, delay)(step)
