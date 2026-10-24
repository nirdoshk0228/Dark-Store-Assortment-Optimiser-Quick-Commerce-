"""
Step 5 - Export the small files the Streamlit app needs (so the app runs and
deploys without the 700 MB raw data).

Run after run_experiment.py:
    python src/export_app_data.py --data data --out outputs --app app/data
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from assortment import basket_greedy, completion_curve, popularity_ranking
from run_experiment import GRID, K, LOCK_SHARE

N_SAMPLE_ORDERS = 3000
MAX_BASKET = 20  # keep simulator baskets readable


def main(data: Path, out: Path, app: Path) -> None:
    app.mkdir(parents=True, exist_ok=True)
    S = sp.load_npz(data / "select.npz")
    T = sp.load_npz(data / "test.npz")
    products = pd.read_csv(data / "products.csv").set_index("product_id")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    alpha = summary["best_alpha"]

    # 1) hybrid assortment for every store size in the grid
    pop = popularity_ranking(S, K)
    rows, check = [], []
    for n in GRID:
        r = basket_greedy(S, n, alpha=alpha, locked=pop[:int(LOCK_SHARE * n)])
        rows.append(pd.DataFrame({"skus": n, "product_id": r}))
        check.append(completion_curve(T, r, [n])["complete"][0])
    pd.concat(rows).to_csv(app / "hybrid_assortments.csv.gz", index=False)
    curves = pd.read_csv(out / "completion_curves.csv")
    assert np.allclose(check, curves["hybrid_complete"]), "hybrid export does not match experiment"

    # 2) a random sample of held-out orders, with product names
    rng = np.random.default_rng(7)
    sizes = np.diff(T.indptr)
    eligible = np.flatnonzero((sizes >= 2) & (sizes <= MAX_BASKET))
    pick = np.sort(rng.choice(eligible, N_SAMPLE_ORDERS, replace=False))
    recs = []
    for i, o in enumerate(pick):
        for p in T.indices[T.indptr[o]:T.indptr[o + 1]]:
            recs.append((i + 1, int(p)))
    orders = pd.DataFrame(recs, columns=["order", "product_id"])
    orders = orders.join(products[["product_name", "aisle", "department"]], on="product_id")
    orders.to_csv(app / "sample_orders.csv.gz", index=False)

    # 3) copy the result files
    for f in ["completion_curves.csv", "summary.json", "alpha_tuning.csv",
              "ranking_popularity.csv", "ranking_basket_greedy.csv"]:
        shutil.copy(out / f, app / f)
    print("app data written to", app)
    for f in sorted(app.iterdir()):
        print(f"  {f.name:32s} {f.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--app", type=Path, default=Path("app/data"))
    a = ap.parse_args()
    main(a.data, a.out, a.app)
