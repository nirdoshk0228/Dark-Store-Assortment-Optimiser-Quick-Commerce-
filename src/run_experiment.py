"""
Step 3 - Popularity vs basket-aware assortment.

1. Choose SKUs on the `select` orders with both methods.
2. Tune the greedy's alpha on the `valid` orders.
3. Report the completion curves on the held-out `test` orders (each user's
   last order) and save rankings + curves for the dashboard.

Usage:  python src/run_experiment.py --data data --out outputs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from assortment import basket_greedy, completion_curve, popularity_ranking

K = 10_000
GRID = list(range(500, K + 1, 500))
ALPHAS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
STORE_SIZE = 5_000  # typical dark store, used for headline numbers
LOCK_SHARE = 0.6    # hybrid: top 60% of slots go to best sellers, rest optimised
                    # (chosen on validation: keeps most of the completion gain
                    #  while losing ~1pp of item coverage)


def main(data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    S = sp.load_npz(data / "select.npz")
    V = sp.load_npz(data / "valid.npz")
    T = sp.load_npz(data / "test.npz")
    products = pd.read_csv(data / "products.csv").set_index("product_id")

    pop = popularity_ranking(S, K)

    # --- tune alpha on validation orders -----------------------------------
    tuning, rankings = [], {}
    for a in ALPHAS:
        rankings[a] = basket_greedy(S, K, alpha=a)
        c = completion_curve(V, rankings[a], GRID)
        tuning.append({"alpha": a, "mean_completion": c["complete"].mean(),
                       f"completion_at_{STORE_SIZE}": c["complete"][GRID.index(STORE_SIZE)]})
        print(f"alpha={a:<5} mean completion over grid {tuning[-1]['mean_completion']:.4f}")
    tuning = pd.DataFrame(tuning)
    tuning.to_csv(out / "alpha_tuning.csv", index=False)
    best_alpha = float(tuning.loc[tuning["mean_completion"].idxmax(), "alpha"])
    greedy = rankings[best_alpha]
    print(f"best alpha = {best_alpha}")

    # --- final evaluation on held-out test orders ---------------------------
    cp = completion_curve(T, pop, GRID)
    cg = completion_curve(T, greedy, GRID)
    # hybrid is re-optimised for every store size N (lock top LOCK_SHARE*N sellers)
    ch = {"complete": [], "max_one_missing": [], "item_coverage": []}
    hybrid_store = None
    for n in GRID:
        r = basket_greedy(S, n, alpha=best_alpha, locked=pop[:int(LOCK_SHARE * n)])
        if n == STORE_SIZE:
            hybrid_store = r
        c = completion_curve(T, r, [n])
        for key in ch:
            ch[key].append(c[key][0])
    ch = {k: np.array(v) for k, v in ch.items()}
    curves = pd.DataFrame({
        "skus": GRID,
        "pop_complete": cp["complete"], "greedy_complete": cg["complete"],
        "pop_max1_missing": cp["max_one_missing"], "greedy_max1_missing": cg["max_one_missing"],
        "pop_item_coverage": cp["item_coverage"], "greedy_item_coverage": cg["item_coverage"],
        "hybrid_complete": ch["complete"], "hybrid_max1_missing": ch["max_one_missing"],
        "hybrid_item_coverage": ch["item_coverage"],
    })
    curves["lift_pp"] = 100 * (curves["greedy_complete"] - curves["pop_complete"])
    curves["lift_rel"] = curves["greedy_complete"] / curves["pop_complete"] - 1
    curves.to_csv(out / "completion_curves.csv", index=False)

    # --- rankings, and what changes at a 5k-SKU store ------------------------
    def label(ids):
        df = products.loc[ids, ["product_name", "aisle", "department"]].reset_index()
        df.insert(0, "rank", np.arange(1, len(ids) + 1))
        return df
    label(pop).to_csv(out / "ranking_popularity.csv", index=False)
    label(greedy).to_csv(out / "ranking_basket_greedy.csv", index=False)
    label(hybrid_store).to_csv(out / f"assortment_hybrid_{STORE_SIZE}.csv", index=False)

    counts = np.asarray(S.sum(axis=0)).ravel()
    pop_set, gr_set = set(pop[:STORE_SIZE]), set(greedy[:STORE_SIZE])
    added = sorted(gr_set - pop_set, key=lambda p: -counts[p])
    dropped = sorted(pop_set - gr_set, key=lambda p: -counts[p])
    swap = lambda ids, tag: products.loc[ids, ["product_name", "aisle", "department"]] \
        .assign(change=tag, orders_in_select=counts[ids]).reset_index()
    pd.concat([swap(added, "added_by_greedy"), swap(dropped, "dropped_by_greedy")]) \
        .to_csv(out / f"swaps_at_{STORE_SIZE}.csv", index=False)

    row = curves.set_index("skus").loc[STORE_SIZE]
    summary = {
        "best_alpha": best_alpha,
        "store_size": STORE_SIZE,
        "test_orders": int(T.shape[0]),
        "popularity_complete": round(float(row["pop_complete"]), 4),
        "greedy_complete": round(float(row["greedy_complete"]), 4),
        "lift_pp": round(float(row["lift_pp"]), 2),
        "lift_relative": round(float(row["lift_rel"]), 4),
        "popularity_item_coverage": round(float(row["pop_item_coverage"]), 4),
        "greedy_item_coverage": round(float(row["greedy_item_coverage"]), 4),
        "skus_swapped": len(added),
        "hybrid_complete": round(float(row["hybrid_complete"]), 4),
        "hybrid_item_coverage": round(float(row["hybrid_item_coverage"]), 4),
        "hybrid_lift_pp": round(100 * float(row["hybrid_complete"] - row["pop_complete"]), 2),
        # SKUs popularity needs to match greedy's completion at STORE_SIZE
        "popularity_skus_needed_to_match": int(next(
            (n for n, v in zip(GRID, cp["complete"]) if v >= row["greedy_complete"]), -1)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(curves.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    a = ap.parse_args()
    main(a.data, a.out)
