"""Leverage sweep for the exploit catalogue: at what max_leverage does the
limited-liability lottery start paying, and when does the covenant actually
fire? Run: uv run --extra dev python scripts/leverage_sweep.py
"""

from statistics import mean

from bazaar_env import core


def sweep(vol: float = 3.0, seeds: range = range(30)) -> None:
    print(f"vol={vol}, seeds={len(seeds)}")
    header = f"{'lev':>4} | {'gambler':>8} {'defaults':>8} | {'taker':>8} {'defaults':>8} | {'passive':>8}"
    print(header)
    for lev in (1.0, 2.0, 3.0, 4.0):
        gambler_returns, gambler_defaults = [], 0
        taker_returns, taker_defaults = [], 0
        passive_returns = []
        for seed in seeds:
            task = core.generate(tier="credit_micro", seed=seed, vol=vol, max_leverage=lev)
            g = core.run_policy(task, core.margin_gambler_policy(seed))
            t = core.run_policy(task, core.margin_taker_policy(1.0))
            p = core.run_policy(task, core.passive_policy)
            gambler_returns.append(g.terminal_return)
            gambler_defaults += int(g.final_state.defaulted)
            taker_returns.append(t.terminal_return)
            taker_defaults += int(t.final_state.defaulted)
            passive_returns.append(p.terminal_return)
        print(
            f"{lev:>4} | {mean(gambler_returns):>8.4f} {gambler_defaults:>8} | "
            f"{mean(taker_returns):>8.4f} {taker_defaults:>8} | {mean(passive_returns):>8.4f}"
        )


if __name__ == "__main__":
    sweep()
