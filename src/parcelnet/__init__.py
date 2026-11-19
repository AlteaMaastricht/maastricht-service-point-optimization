"""parcelnet: service-point / parcel-locker network design on a road graph.

Cleaned, packaged version of a 2025 team consulting case (see README and
docs/PROVENANCE.md). Modules:

- graph        road graph construction and all-pairs shortest paths (scipy Dijkstra)
- instance     problem instance container, course-format loader, demand estimation
- toy          small synthetic grid instance (used by tests and the example)
- optimizer    simulated-annealing facility-location heuristic (SP-only or SP+APL)
- sensitivity  one-at-a-time parameter sensitivity and robustness runs
- plotting     static solution map
"""

__all__ = ["graph", "instance", "toy", "optimizer", "sensitivity", "plotting"]
__version__ = "0.1.0"
