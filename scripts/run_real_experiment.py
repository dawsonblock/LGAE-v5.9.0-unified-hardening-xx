#!/usr/bin/env python3
"""Real-world experiment: does LGAE governance improve a real downstream task?

This is the experiment the v5.x releases were missing.  It uses a **real**
graph with **real** ground truth (Zachary's Karate Club — a social network
with two known communities, shipped with NetworkX so no external download is
required) and a **real** downstream objective: recovering the two
communities from a latent embedding via clustering.

It compares four conditions on the same graph:

    - raw            : spectral embedding of the unmodified graph (baseline)
    - random_add     : add k random edges, then embed
    - spectral_heur  : add edges the spectral heuristic picks, then embed
    - lgae_governed  : run the LGAE engine propose->govern->commit loop for
                       k steps, then embed the engine's latent

Metric: clustering accuracy vs the ground-truth community labels, with
label permutation matched so the score is invariant to which cluster is
called "0".

This is a small, real-world signal -- not a benchmark-suite claim.  Karate
Club has 34 nodes, so variance is high; treat the result as a sanity check
that the governance loop does not *harm* a real downstream task and
ideally improves it over the raw baseline.

Usage::

    python scripts/run_real_experiment.py
    python scripts/run_real_experiment.py --steps 8 --seed 0 --out real_experiment.json

Exit code 0 if lgae_governed >= raw (governance does not harm the task),
else 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import random

import networkx as nx
import numpy as np
import torch
from scipy.cluster.vq import kmeans2

from lgae_v3 import LGAEConfig, LGAEEngine, make_graph_buffers
from lgae_v3.benchmark.baselines import SpectralHeuristicController
from lgae_v3.benchmark.tasks import StructuralAction
from lgae_v3.operators import spectral_gap_graphbuffers


def _karate_with_labels() -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Return (graph, node_order, ground_truth_labels)."""
    G = nx.karate_club_graph()
    order = sorted(G.nodes())
    # "Mr. Hi" club vs "Officer" club.
    labels = np.array([0 if G.nodes[n]["club"] == "Mr. Hi" else 1 for n in order])
    return G, np.array(order), labels


def _spectral_embedding(graph: nx.Graph, dim: int = 8) -> np.ndarray:
    """Laplacian eigenmap embedding (bottom non-trivial eigenvectors)."""
    n = graph.number_of_nodes()
    order = sorted(graph.nodes())
    A = nx.to_numpy_array(graph, nodelist=order)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    Dinv = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(n) - Dinv @ A @ Dinv
    w, V = np.linalg.eigh(L)
    # Skip the trivial zero eigenvalue; take the next `dim`.
    return V[:, 1:1 + dim]


def _cluster_accuracy(embedding: np.ndarray, labels: np.ndarray, k: int = 2, seed: int = 0) -> float:
    """k-means clustering accuracy with label permutation matching."""
    if embedding.shape[1] == 0:
        return 0.0
    np.random.seed(seed)
    centroids, pred = kmeans2(embedding, k, minit="++", seed=seed)
    # Match predicted labels to ground truth via majority vote per cluster.
    correct = 0
    for c in range(k):
        mask = pred == c
        if not mask.any():
            continue
        majority = np.bincount(labels[mask], minlength=k).argmax()
        correct += int((labels[mask] == majority).sum())
    acc = correct / max(len(labels), 1)
    # Clustering is label-invariant; report the better of the two labelings.
    return max(acc, 1.0 - acc) if k == 2 else acc


def _graph_to_buffers(graph: nx.Graph, capacity_factor: float = 2.0):
    order = sorted(graph.nodes())
    idx = {n: i for i, n in enumerate(order)}
    edges = [(idx[u], idx[v], 1.0) for u, v in graph.edges()]
    n = len(order)
    cap = max(len(edges) + 16, int(len(edges) * capacity_factor))
    return make_graph_buffers(n, edges, capacity=cap), n


def _add_random_edges(graph: nx.Graph, k: int, rng: random.Random) -> nx.Graph:
    g = graph.copy()
    nodes = list(g.nodes())
    existing = set(g.edges())
    added = 0
    tries = 0
    while added < k and tries < 10 * k:
        u, v = rng.sample(nodes, 2)
        if (u, v) not in existing and (v, u) not in existing:
            g.add_edge(u, v); existing.add((u, v)); added += 1
        tries += 1
    return g


def _add_heuristic_edges(graph: nx.Graph, k: int, z: np.ndarray) -> nx.Graph:
    """Add edges between non-adjacent nodes that are close in the embedding."""
    g = graph.copy()
    order = sorted(g.nodes())
    n = len(order)
    existing = set(g.edges())
    # Pairwise embedding distance for non-edges.
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            if (order[i], order[j]) not in existing:
                d = float(np.linalg.norm(z[i] - z[j]))
                cand.append((d, order[i], order[j]))
    cand.sort()
    for _, u, v in cand[:k]:
        g.add_edge(u, v)
    return g


def _run_lgae_governed(graph: nx.Graph, steps: int, seed: int) -> tuple[nx.Graph, np.ndarray]:
    """Run the LGAE engine propose->govern->commit loop, return final graph + latent."""
    buffers, n = _graph_to_buffers(graph)
    cfg = LGAEConfig()
    cfg.seed = seed
    eng = LGAEEngine(buffers, cfg)
    for _ in range(steps):
        eng.diffuse_(eta=0.02)
        eng.fiber_tick()
        mut = eng.propose_midpoint_edge()
        if mut:
            eng.evaluate_and_maybe_commit(mut)
    # Final latent from the engine's fiber state.
    z = eng.fibers.latent.detach().cpu().numpy()
    # Reconstruct the networkx graph from the (possibly mutated) buffers.
    final = nx.Graph()
    final.add_nodes_from(range(n))
    valid = eng.graph.valid.bool()
    src = eng.graph.src[valid].tolist()
    dst = eng.graph.dst[valid].tolist()
    for u, v in zip(src, dst):
        final.add_edge(int(u), int(v))
    return final, z


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8, help="LGAE governance steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=8, help="embedding dimension")
    p.add_argument("--added-edges", type=int, default=6, help="edges to add for random/heuristic")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    G, order, labels = _karate_with_labels()
    n = len(order)

    results: dict[str, dict] = {}

    # 1. Raw baseline.
    z_raw = _spectral_embedding(G, dim=args.dim)
    acc_raw = _cluster_accuracy(z_raw, labels, seed=args.seed)
    lam_raw = float(spectral_gap_graphbuffers(_graph_to_buffers(G)[0])[0])
    results["raw"] = {"accuracy": acc_raw, "spectral_gap": lam_raw, "edges": G.number_of_edges()}

    # 2. Random edge addition.
    G_rand = _add_random_edges(G, args.added_edges, rng)
    z_rand = _spectral_embedding(G_rand, dim=args.dim)
    acc_rand = _cluster_accuracy(z_rand, labels, seed=args.seed)
    lam_rand = float(spectral_gap_graphbuffers(_graph_to_buffers(G_rand)[0])[0])
    results["random_add"] = {"accuracy": acc_rand, "spectral_gap": lam_rand, "edges": G_rand.number_of_edges()}

    # 3. Spectral-heuristic edge addition (add edges between close non-adjacent nodes).
    G_heur = _add_heuristic_edges(G, args.added_edges, z_raw)
    z_heur = _spectral_embedding(G_heur, dim=args.dim)
    acc_heur = _cluster_accuracy(z_heur, labels, seed=args.seed)
    lam_heur = float(spectral_gap_graphbuffers(_graph_to_buffers(G_heur)[0])[0])
    results["spectral_heuristic"] = {"accuracy": acc_heur, "spectral_gap": lam_heur, "edges": G_heur.number_of_edges()}

    # 4. LGAE governed.
    try:
        G_lgae, z_lgae = _run_lgae_governed(G, steps=args.steps, seed=args.seed)
        # Cluster on the engine's own latent (its fiber state), padded/truncated to dim.
        if z_lgae.shape[1] < args.dim:
            z_lgae = np.pad(z_lgae, ((0, 0), (0, args.dim - z_lgae.shape[1])))
        else:
            z_lgae = z_lgae[:, :args.dim]
        acc_lgae = _cluster_accuracy(z_lgae, labels, seed=args.seed)
        lam_lgae = float(spectral_gap_graphbuffers(_graph_to_buffers(G_lgae)[0])[0])
        results["lgae_governed"] = {"accuracy": acc_lgae, "spectral_gap": lam_lgae, "edges": G_lgae.number_of_edges()}
    except Exception as e:
        results["lgae_governed"] = {"error": str(e)}

    # Summary.
    print(f"Karate Club community recovery (n={n}, ground truth = 2 communities)")
    print(f"{'condition':22s} {'accuracy':>10s} {'lambda2':>10s} {'edges':>8s}")
    print("-" * 52)
    for cond in ["raw", "random_add", "spectral_heuristic", "lgae_governed"]:
        r = results[cond]
        if "error" in r:
            print(f"{cond:22s} {'ERROR':>10s} {'-':>10s} {'-':>8s}  {r['error']}")
        else:
            print(f"{cond:22s} {r['accuracy']:10.4f} {r['spectral_gap']:10.4f} {r['edges']:8d}")

    payload = {
        "schema": "LGAE_REAL_EXPERIMENT",
        "dataset": "karate_club",
        "n_nodes": n,
        "n_ground_truth_communities": 2,
        "seed": args.seed,
        "steps": args.steps,
        "added_edges": args.added_edges,
        "embedding_dim": args.dim,
        "results": results,
        "note": "Small real-world sanity check. Karate Club has 34 nodes; treat as directional signal, not a benchmark-suite claim.",
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
    else:
        print("\n" + json.dumps(payload, indent=2))

    lgae_acc = results["lgae_governed"].get("accuracy", 0.0)
    return 0 if lgae_acc >= acc_raw else 1


if __name__ == "__main__":
    raise SystemExit(main())
