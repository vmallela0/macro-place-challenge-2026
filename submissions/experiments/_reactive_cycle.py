"""Reactive cycle scheduler — picks the phase that gained most last time.

Each cycle, run ALL phases briefly. Track which phase contributed most.
Next cycle, allocate more budget to the winner.

Self-tuning cycle schedule.
"""
import time
import numpy as np


class ReactiveCycler:
    def __init__(self, phases):
        """phases: list of (name, callable). Callable signature:
        (pos_np, benchmark, incr_eval, budget) -> (pos, new_cost)"""
        self.phases = phases
        self.credits = {name: 1.0 for name, _ in phases}  # initial equal weight
        self.contribution = {name: 0.0 for name, _ in phases}

    def run_cycle(self, pos_np, benchmark, incr_eval, total_budget, verbose=False):
        # Allocate budget proportional to credits
        total_cred = sum(self.credits.values())
        budgets = {name: total_budget * self.credits[name] / total_cred
                   for name, _ in self.phases}

        cost_before = incr_eval.get_proxy_cost()
        phase_gains = {}

        for name, fn in self.phases:
            pre = incr_eval.get_proxy_cost()
            try:
                pos_np, new_cost = fn(pos_np, benchmark, incr_eval, budgets[name])
            except Exception as e:
                if verbose:
                    print(f"    {name} error: {e}")
                continue
            post = incr_eval.get_proxy_cost()
            gain = pre - post
            phase_gains[name] = gain

        # Update credits: EWMA
        alpha = 0.3
        for name in phase_gains:
            # Normalize credit update by gain
            self.contribution[name] = (1 - alpha) * self.contribution[name] + alpha * max(0, phase_gains[name])
            self.credits[name] = 0.1 + self.contribution[name]  # floor 0.1

        if verbose:
            gains_str = ", ".join(f"{k}={v:+.4f}" for k, v in phase_gains.items())
            print(f"    reactive: {gains_str}")
        return pos_np, incr_eval.get_proxy_cost()
