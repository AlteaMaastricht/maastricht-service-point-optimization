"""Static solution map. Port of ``SA5.visualize_solution_static`` (SA5.py:1539-1598).

Fix: the current SA5.py built the APL list from the boolean ``apl_open``
vector (``opened_apls = [loc for i, loc in enumerate(apl_open) if loc ...]``),
so it plotted node ``True == 1`` instead of the locker nodes. Here the open
lockers' node ids are used. As in the original, "TotCap" in the title is the
service-point capacity only (APL capacity is not included).
"""
from __future__ import annotations

from typing import Optional

from .optimizer import ServicePointOptimizer


def plot_solution(opt: ServicePointOptimizer, save_path: Optional[str] = None, annotate: bool = True):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sol = opt.best_solution
    s = opt.summarize(sol)
    inst = opt.inst
    xy = inst.coords
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=8, alpha=0.6, label="All nodes", zorder=1)
    if s["open_apl_nodes"]:
        idx = [inst.node_to_idx[n] for n in s["open_apl_nodes"]]
        ax.scatter(xy[idx, 0], xy[idx, 1], c="blue", s=80, alpha=0.8,
                   label=f"APLs ({len(idx)})", zorder=3)
    sp_cols = [j for j in range(opt.n_sp) if sol["y"][j]]
    idx = [opt.sp_nodes[j] for j in sp_cols]
    ax.scatter(xy[idx, 0], xy[idx, 1], c="red", s=100, alpha=0.9, label=f"SPs ({len(idx)})", zorder=4)
    if annotate:
        for j, n in zip(sp_cols, idx):
            ax.annotate(f"{int(inst.node_ids[n])}\nCap:{sol['capacity'][j]:.0f}", (xy[n, 0], xy[n, 1]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=5)
    ax.set_title(f"Service-point network solution\n{s['n_service_points']} SPs, {s['n_apls']} APLs, "
                 f"SP capacity: {s['sp_capacity_per_day']:,.0f} parcels/day")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
