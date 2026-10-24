"""
Step 1 - Data prep for the dark-store assortment project.

Builds three disjoint sets of orders from the Instacart Market Basket data:
  * select : random sample of prior orders  -> used to CHOOSE the assortment
  * valid  : another random sample of prior -> used to TUNE the algorithm
  * test   : each user's final order ("train" eval_set) -> used to REPORT results
               (these happen after every prior order, so there is no look-ahead)

Each set is saved as a sparse order x product matrix (scipy CSR, binary).

Usage:  python src/prep.py --raw <folder with the Instacart CSVs> --out data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 42


def to_matrix(lines: pd.DataFrame, n_products: int) -> sp.csr_matrix:
    """Order-line table -> binary CSR matrix (rows = orders, cols = product_id)."""
    codes, _ = pd.factorize(lines["order_id"])
    m = sp.csr_matrix(
        (np.ones(len(lines), dtype=np.int8), (codes, lines["product_id"].to_numpy())),
        shape=(codes.max() + 1, n_products),
    )
    m.data[:] = 1  # guard against duplicate lines
    m.sort_indices()
    return m


def main(raw: Path, out: Path, n_select: int, n_valid: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dtypes = {"order_id": "int32", "product_id": "int32"}

    products = pd.read_csv(raw / "products.csv")
    aisles = pd.read_csv(raw / "aisles.csv")
    depts = pd.read_csv(raw / "departments.csv")
    products = products.merge(aisles, on="aisle_id").merge(depts, on="department_id")
    n_products = int(products["product_id"].max()) + 1
    products.to_csv(out / "products.csv", index=False)

    prior = pd.read_csv(raw / "order_products__prior.csv",
                        usecols=["order_id", "product_id"], dtype=dtypes)
    test = pd.read_csv(raw / "order_products__train.csv",
                       usecols=["order_id", "product_id"], dtype=dtypes)

    rng = np.random.default_rng(SEED)
    prior_ids = prior["order_id"].unique()
    rng.shuffle(prior_ids)
    sel_ids = prior_ids[:n_select]
    val_ids = prior_ids[n_select:n_select + n_valid]

    sets = {
        "select": prior[prior["order_id"].isin(sel_ids)],
        "valid": prior[prior["order_id"].isin(val_ids)],
        "test": test,
    }
    for name, lines in sets.items():
        m = to_matrix(lines, n_products)
        sp.save_npz(out / f"{name}.npz", m)
        sizes = np.diff(m.indptr)
        print(f"{name:6s}: {m.shape[0]:>9,} orders | {m.nnz:>11,} lines | "
              f"median basket {np.median(sizes):.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--n-select", type=int, default=1_000_000)
    ap.add_argument("--n-valid", type=int, default=200_000)
    a = ap.parse_args()
    main(a.raw, a.out, a.n_select, a.n_valid)
