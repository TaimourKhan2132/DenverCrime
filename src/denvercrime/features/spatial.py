"""H3 hexagon helpers (h3-py v4 API)."""

from __future__ import annotations

from collections.abc import Iterable

import h3
import numpy as np
import pandas as pd


def to_cells(lat: Iterable[float], lon: Iterable[float], resolution: int) -> np.ndarray:
    return np.array([h3.latlng_to_cell(a, b, resolution) for a, b in zip(lat, lon)], dtype=object)


def cell_table(cells: Iterable[str]) -> pd.DataFrame:
    """Centroid and area for each cell, indexed by cell id."""
    cells = list(cells)
    centroids = [h3.cell_to_latlng(c) for c in cells]
    return pd.DataFrame(
        {
            "cell_lat": [lat for lat, _ in centroids],
            "cell_lon": [lon for _, lon in centroids],
            "area_km2": [h3.cell_area(c, unit="km^2") for c in cells],
        },
        index=pd.Index(cells, name="cell"),
    )


def neighbor_matrix(cells: list[str], ring: int) -> np.ndarray:
    """Row-normalised adjacency: entry (i, j) = 1/deg(i) if cell j is within `ring` of cell i.

    Only neighbours inside `cells` count, so edge cells average over fewer neighbours.
    Cells with no neighbours in the set get an all-zero row.
    """
    position = {c: i for i, c in enumerate(cells)}
    adj = np.zeros((len(cells), len(cells)), dtype=np.float64)
    for i, cell in enumerate(cells):
        for other in h3.grid_disk(cell, ring):
            j = position.get(other)
            if j is not None and j != i:
                adj[i, j] = 1.0
    degree = adj.sum(axis=1, keepdims=True)
    return np.divide(adj, degree, out=np.zeros_like(adj), where=degree > 0)


def cell_polygon(cell: str) -> list[list[float]]:
    """Closed GeoJSON ring ([lon, lat] pairs) for a cell."""
    ring = [[lon, lat] for lat, lon in h3.cell_to_boundary(cell)]
    return ring + [ring[0]]
