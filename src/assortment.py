"""
Step 2 - Two ways to choose which SKUs a dark store stocks.

1. popularity_ranking : stock the products that appear in the most orders.
2. basket_greedy      : add products one at a time; each step pick the product
                        that does the most to COMPLETE baskets.

How basket_greedy scores a product
----------------------------------
Every order (basket) tracks how many of its items are still missing, m.
A missing item in that basket earns the basket's weight  w(m) = alpha ** (m - 1):
    m = 1 (this is the last missing item) -> weight 1   (adding it completes the order)
    m = 2                                 -> weight alpha
    m = 3                                 -> weight alpha^2 ...
A product's score is the sum of weights over all baskets it appears in.
The highest-scoring product is stocked, the baskets that contain it drop to
m - 1, and the scores of every product in those baskets are updated.

alpha = 1 just counts incomplete baskets (close to popularity);
small alpha focuses hard on baskets that are nearly complete.
"""
import numpy as np
import scipy.sparse as sp


def popularity_ranking(orders: sp.csr_matrix, k: int) -> np.ndarray:
    counts = np.asarray(orders.sum(axis=0)).ravel()
    return np.argsort(-counts, kind="stable")[:k]


def _gather(indptr, indices, rows):
    """Column indices of all non-zeros in `rows`, plus which row each came from."""
    starts, ends = indptr[rows], indptr[rows + 1]
    lens = ends - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    # position of each element = its row start + offset within the row
    row_of = np.repeat(np.arange(len(rows)), lens)
    offs = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    return indices[starts[row_of] + offs], row_of


def basket_greedy(orders: sp.csr_matrix, k: int, alpha: float = 0.3,
                  locked=None, verbose: bool = False) -> np.ndarray:
    """`locked`: optional list of SKUs that are always stocked first
    (e.g. the top sellers); the greedy then fills the remaining slots."""
    locked = [] if locked is None else [int(x) for x in locked]
    csr = orders.tocsr()
    csc = orders.tocsc()
    n_items = orders.shape[1]
    missing = np.diff(csr.indptr).astype(np.int64)  # items still missing per basket

    def weight(m):
        return np.where(m > 0, alpha ** (np.maximum(m, 1) - 1), 0.0)

    w = weight(missing)
    # initial score: sum of basket weights over baskets containing each item
    score = np.asarray(csr.T @ w).ravel().astype(np.float64)
    chosen = np.zeros(n_items, bool)
    picks = []

    for step in range(k):
        if step < len(locked):
            j = locked[step]
        else:
            s = np.where(chosen, -np.inf, score)
            j = int(np.argmax(s))
            if not np.isfinite(s[j]) or s[j] <= 0:
                break
        picks.append(j)
        chosen[j] = True

        rows = csc.indices[csc.indptr[j]:csc.indptr[j + 1]]
        if len(rows) == 0:
            continue
        old = missing[rows]
        missing[rows] = old - 1
        delta = weight(old - 1) - weight(old)
        items, row_of = _gather(csr.indptr, csr.indices, rows)
        score += np.bincount(items, weights=delta[row_of], minlength=n_items)

        if verbose and (step + 1) % 1000 == 0:
            print(f"  picked {step + 1:,} SKUs | baskets complete in select set: "
                  f"{(missing == 0).mean():.3f}")
    return np.array(picks)


def completion_curve(orders: sp.csr_matrix, ranking: np.ndarray, grid) -> dict:
    """For each N in grid: share of orders fully covered by the first N SKUs,
    and share with at most one item missing."""
    n_items = orders.shape[1]
    rank = np.full(n_items, np.iinfo(np.int64).max // 2, dtype=np.int64)
    rank[ranking] = np.arange(len(ranking))
    csr = orders.tocsr()
    r = rank[csr.indices]
    sizes = np.diff(csr.indptr)
    # largest and second-largest rank in each basket
    order_of = np.repeat(np.arange(csr.shape[0]), sizes)
    srt = np.lexsort((r, order_of))
    r_sorted = r[srt]
    ends = csr.indptr[1:] - 1
    max_r = r_sorted[ends]
    second = np.where(sizes > 1, r_sorted[np.maximum(ends - 1, 0)], -1)
    full = np.array([(max_r < n).mean() for n in grid])
    one_missing = np.array([(second < n).mean() for n in grid])
    # share of item lines covered
    lines = np.array([(r < n).mean() for n in grid])
    return {"grid": np.asarray(grid), "complete": full,
            "max_one_missing": one_missing, "item_coverage": lines}
