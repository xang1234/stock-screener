"""Per-market rate-budget load/soak harness (bead asia.9.3).

Measures wall-clock completion, 429 incident count, batch tail-latency, and
worker memory/CPU across a simulated 4-market parallel weekly refresh. Snapshots
results to JSON for regression-gated CI comparison.
"""
