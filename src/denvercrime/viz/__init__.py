"""Figures. Each `plot_*` returns an open matplotlib Figure, and saves it when given a path."""

from __future__ import annotations

from pathlib import Path


def finish_figure(fig, out_path: Path | None = None):
    """Tidy a figure's layout, save it when `out_path` is given, and return it still open."""
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig
