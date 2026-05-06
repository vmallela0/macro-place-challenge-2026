"""Treewidth upper bound via min-degree elimination heuristic.

Treewidth measures how "tree-like" a graph is. If the netlist's
2-clique expansion has treewidth ≤ t, exact L1 HPWL placement is
solvable in O(N · 2^t) time via tree-decomposition dynamic programming
(Bodlaender-Koster). For logic-synthesis netlists, treewidth has been
reported in the 8-40 range (Marinescu-Dechter 2009). IBM ICCAD-04
bench netlists have not been measured publicly.

We compute an UPPER BOUND on treewidth via the **min-degree elimination
heuristic**: at each step, eliminate the minimum-degree vertex,
turning its neighborhood into a clique. The maximum degree at
elimination time is an upper bound on treewidth (Bodlaender 1996).

Min-degree is fast (O(V·E·log V) with a heap) and known to be within
a constant factor of optimal on many real graphs. For tighter bounds
one can use min-fill (which adds the fewest fill-in edges), but min-
degree is the standard sanity check.

Hypergraph → graph reduction
----------------------------
Each net of size k becomes a k-clique in the 2-graph (union of pairwise
edges). For very-high-degree nets (k > THRESH=20), the clique blows up
quadratically and dominates the treewidth. Such nets typically are
clock or power signals not relevant for HPWL anyway. We drop nets with
k > MAX_NET_SIZE (default 30) and report how many were dropped.

Usage
-----
    .venv/bin/python research/lower_bounds/treewidth.py --benchmark ibm01
"""
from __future__ import annotations
import argparse
import heapq
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))

from macro_place.benchmark import Benchmark


def build_clique_graph(net_nodes_lists, n_nodes, max_net_size=30):
    """Return adj[node] = set of neighbors. Drop nets of size > max_net_size
    (clock/power-like; they expand to k² edges and are not typically HPWL-
    bottlenecked).
    """
    adj = [set() for _ in range(n_nodes)]
    n_dropped = 0
    n_total_nets = 0
    for net in net_nodes_lists:
        nodes = list(set(int(n) for n in net))   # dedup pins on same node
        n_total_nets += 1
        if len(nodes) > max_net_size:
            n_dropped += 1
            continue
        # add pairwise edges
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if u == v:
                    continue
                adj[u].add(v)
                adj[v].add(u)
    return adj, n_dropped, n_total_nets


def min_fill_elimination(adj, verbose=False, max_n=2000):
    """Min-fill: at each step, eliminate the vertex that adds the fewest
    fill-in edges (edges between non-adjacent neighbors needed to make
    them a clique). Tighter upper bound than min-degree on most graphs
    but O(V³) per step worst-case. Skipped if n > max_n.
    """
    n = len(adj)
    if n > max_n:
        return None, None, None, None  # skip; too slow
    adj = [set(s) for s in adj]
    deleted = [False] * n
    tw_ub = 0
    elim_count = 0
    fill_in_total = 0
    t0 = time.time()
    for _ in range(n):
        # Find vertex with min fill-in.
        best_v = -1
        best_fill = float("inf")
        best_deg = -1
        for v in range(n):
            if deleted[v]:
                continue
            neigh = adj[v]
            d = len(neigh)
            # count missing edges among neigh
            fill = 0
            neigh_list = list(neigh)
            for i, u in enumerate(neigh_list):
                au = adj[u]
                for w in neigh_list[i + 1:]:
                    if w not in au:
                        fill += 1
            if fill < best_fill or (fill == best_fill and d < best_deg):
                best_v = v
                best_fill = fill
                best_deg = d
        if best_v < 0:
            break
        v = best_v
        tw_ub = max(tw_ub, best_deg)
        neigh = list(adj[v])
        for i, u in enumerate(neigh):
            for w in neigh[i + 1:]:
                if w not in adj[u]:
                    adj[u].add(w)
                    adj[w].add(u)
                    fill_in_total += 1
        for u in neigh:
            adj[u].discard(v)
        adj[v] = set()
        deleted[v] = True
        elim_count += 1
        if verbose and elim_count % 200 == 0:
            print(f"    [tw-fill] eliminated {elim_count}/{n}; tw_ub={tw_ub}; "
                  f"fill={fill_in_total}; wall {time.time()-t0:.1f}s",
                  flush=True)
    return tw_ub, elim_count, fill_in_total, time.time() - t0


def min_degree_elimination(adj, verbose=False):
    """Returns (treewidth_upper_bound, elimination_history).

    Each step:
      1. Find v with minimum degree.
      2. tw_ub = max(tw_ub, deg(v))
      3. Remove v; add edges among v's neighbors to make them a clique.
    """
    n = len(adj)
    # Use a fresh adjacency we can mutate.
    adj = [set(s) for s in adj]
    # Heap entries: (degree, node). We allow stale entries (recheck on pop).
    heap = [(len(adj[v]), v) for v in range(n) if adj[v] is not None]
    heapq.heapify(heap)
    deleted = [False] * n
    tw_ub = 0
    elim_count = 0
    fill_in_total = 0
    t0 = time.time()
    while heap:
        deg, v = heapq.heappop(heap)
        if deleted[v]:
            continue
        if deg != len(adj[v]):
            heapq.heappush(heap, (len(adj[v]), v))
            continue
        # Eliminate v.
        tw_ub = max(tw_ub, deg)
        neigh = list(adj[v])
        # Make neigh into a clique.
        for i, u in enumerate(neigh):
            for w in neigh[i + 1:]:
                if w not in adj[u]:
                    adj[u].add(w)
                    adj[w].add(u)
                    fill_in_total += 1
        # Disconnect v.
        for u in neigh:
            adj[u].discard(v)
        adj[v] = set()
        deleted[v] = True
        elim_count += 1
        # Re-push neighbors with their new degrees.
        for u in neigh:
            heapq.heappush(heap, (len(adj[u]), u))
        if verbose and elim_count % 1000 == 0:
            print(f"    [tw] eliminated {elim_count}/{n}; "
                  f"current tw_ub={tw_ub}; fill={fill_in_total}; "
                  f"wall {time.time()-t0:.1f}s",
                  flush=True)
    return tw_ub, elim_count, fill_in_total, time.time() - t0


def _load_netlist_from_plc(bench_name):
    """Use IncrementalEvaluator to build the flat (pin_macro, net_starts)
    arrays. Returns (pin_node_ids, net_starts, n_nodes) where pin_node_ids
    has macro indices for macro-pins and unique negative-index ids for
    port pins (so each port pin is its own node).
    """
    import importlib.util as ilu
    import torch as _t
    v1_spec = ilu.spec_from_file_location(
        "_v1_tw", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1)
    pt_path = str(ROOT / "benchmarks" / "processed" / "public" /
                   f"{bench_name}.pt")
    raw = _t.load(pt_path, weights_only=False)
    valid_keys = set(Benchmark.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    if "num_hard_macros" not in filtered:
        filtered["num_hard_macros"] = filtered["num_macros"]
        filtered["num_soft_macros"] = 0
    if "soft_macro_indices" not in filtered:
        filtered["soft_macro_indices"] = []
    if "port_positions" not in filtered:
        filtered["port_positions"] = _t.zeros(0, 2)
    if "macro_pin_offsets" not in filtered:
        filtered["macro_pin_offsets"] = []
    bench = Benchmark(**filtered)
    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    pin_macro = np.asarray(incr.pin_macro)
    net_starts = np.asarray(incr.net_starts)
    n_macros = bench.num_macros
    # Re-index ports: each port pin gets a unique node id ≥ n_macros.
    pin_node = pin_macro.copy().astype(np.int64)
    next_port_id = n_macros
    for i in range(pin_node.shape[0]):
        if pin_node[i] < 0:
            pin_node[i] = next_port_id
            next_port_id += 1
    n_nodes = int(next_port_id)
    # Build per-net node lists.
    net_node_lists = []
    for nid in range(net_starts.shape[0] - 1):
        nodes = pin_node[net_starts[nid]:net_starts[nid + 1]]
        net_node_lists.append(np.unique(nodes))   # dedup pins on same node
    return bench, net_node_lists, n_nodes


def measure_benchmark(bench_name, max_net_size=30, verbose=False):
    bench, net_node_lists, n_total_nodes = _load_netlist_from_plc(bench_name)
    print(f"  {bench}", flush=True)
    print(f"  total nodes (macros + port-pins): {n_total_nodes}", flush=True)

    adj, n_dropped, n_total_nets = build_clique_graph(
        net_node_lists, n_total_nodes, max_net_size=max_net_size)
    n_edges = sum(len(s) for s in adj) // 2
    avg_deg = (2 * n_edges) / n_total_nodes if n_total_nodes else 0.0
    max_deg = max(len(s) for s in adj) if adj else 0
    print(f"  clique-graph: {n_edges} edges, avg deg={avg_deg:.2f}, "
          f"max deg={max_deg}", flush=True)
    print(f"  dropped {n_dropped}/{n_total_nets} nets of size > "
          f"{max_net_size}", flush=True)

    tw_md, elim_count, fill_md, wall_md = min_degree_elimination(
        adj, verbose=verbose)
    print(f"  min-degree: tw_ub={tw_md}, fill={fill_md}, wall={wall_md:.2f}s",
          flush=True)

    # min-fill is O(V³); only run for very small graphs as a sanity tighten
    tw_mf, _, fill_mf, wall_mf = min_fill_elimination(
        adj, verbose=verbose, max_n=int(__import__("os").environ.get("MAX_MF_N", "1400")))
    if tw_mf is not None:
        print(f"  min-fill:   tw_ub={tw_mf}, fill={fill_mf}, "
              f"wall={wall_mf:.2f}s", flush=True)
    else:
        print(f"  min-fill:   skipped (n>{2000})", flush=True)
        wall_mf = 0.0

    tw_ub = min(tw_md, tw_mf if tw_mf is not None else tw_md)
    return {
        "bench": bench_name,
        "n_macros": int(bench.num_macros),
        "n_hard": int(bench.num_hard_macros),
        "n_ports": int(bench.port_positions.shape[0]),
        "n_nets": int(bench.num_nets),
        "n_dropped_large_nets": n_dropped,
        "n_edges": n_edges,
        "max_degree": max_deg,
        "treewidth_ub_md": tw_md,
        "treewidth_ub_mf": tw_mf if tw_mf is not None else "N/A",
        "treewidth_ub": tw_ub,
        "fill_in_md": fill_md,
        "fill_in_mf": fill_mf if fill_mf is not None else "N/A",
        "wall_s_md": round(wall_md, 2),
        "wall_s_mf": round(wall_mf, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="all",
                     help="Bench name or 'all' for ibm01..ibm18")
    ap.add_argument("--max_net_size", type=int, default=30)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.benchmark == "all":
        benches = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07",
                   "ibm08", "ibm09", "ibm10", "ibm11", "ibm12", "ibm13",
                   "ibm14", "ibm15", "ibm16", "ibm17", "ibm18"]
    else:
        benches = [args.benchmark]

    results = []
    for b in benches:
        print(f"\n=== {b} ===", flush=True)
        r = measure_benchmark(b, max_net_size=args.max_net_size,
                                verbose=args.verbose)
        results.append(r)
        print(f"  ► treewidth_ub={r['treewidth_ub']} "
              f"(md={r['treewidth_ub_md']}, mf={r['treewidth_ub_mf']})",
              flush=True)

    # Print summary
    print("\n=== Summary ===")
    print(f"{'bench':<8} {'macros':>7} {'nets':>6} {'edges':>7} "
          f"{'maxdeg':>7} {'tw_md':>6} {'tw_mf':>6}")
    for r in results:
        print(f"{r['bench']:<8} {r['n_macros']:>7} {r['n_nets']:>6} "
              f"{r['n_edges']:>7} {r['max_degree']:>7} "
              f"{r['treewidth_ub_md']:>6} {str(r['treewidth_ub_mf']):>6}")

    out_csv = ROOT / "research" / "lower_bounds" / "treewidth_results.csv"
    with open(out_csv, "w") as f:
        keys = ["bench", "n_macros", "n_hard", "n_ports", "n_nets",
                "n_dropped_large_nets", "n_edges", "max_degree",
                "treewidth_ub_md", "treewidth_ub_mf", "treewidth_ub",
                "fill_in_md", "fill_in_mf",
                "wall_s_md", "wall_s_mf"]
        f.write(",".join(keys) + "\n")
        for r in results:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nresults → {out_csv}")


if __name__ == "__main__":
    main()
