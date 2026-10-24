# Dark-Store Assortment Optimiser

**Question:** A Blinkit-style dark store holds ~5,000 SKUs. Which ones should it stock?

**Finding:** Stocking the best sellers maximises *items* covered, not *orders* completed.
In quick commerce one missing item can lose the whole order. A basket-aware selection
completes **15% more orders from the same shelf space** (30.7% → 35.3% of held-out orders
fully in stock at 5,000 SKUs). Popularity ranking needs ~1,000 extra SKUs to match it.

| At 5,000 SKUs (131,209 held-out orders) | Popularity | Basket-aware | Hybrid |
|---|---|---|---|
| Orders fully in stock | 30.7% | **35.3%** | 34.4% |
| Orders with ≤1 item missing | 54.5% | 55.1% | **55.4%** |
| Items ordered that are in stock | **79.7%** | 74.5% | 78.0% |

**Trade-off:** pure basket-aware drops some staples (e.g. ketchup, cream cheese) for niche
items that complete small baskets, so item coverage falls ~5 pp. The **hybrid** keeps the
top 60% of slots for best sellers and optimises the rest: +3.7 pp complete orders for only
−1.7 pp item coverage. This is the version I'd recommend.

## Method
1. **Data:** Instacart Market Basket (3.4M orders, 49.7k products).
   * `select` = 1M random past orders → choose the assortment
   * `valid` = 200k other past orders → tune the algorithm
   * `test` = each user's latest order (131k) → report results; these come after all past orders, so there's no leakage
2. **Popularity baseline:** rank SKUs by number of orders.
3. **Basket-aware greedy:** add SKUs one at a time. Each order with `m` items still missing
   gives each missing item a weight `α^(m−1)` (the last missing item counts 1, the second-to-last α, …).
   Stock the SKU with the highest total weight, update, repeat. α tuned on `valid` (best: 0.05).
4. **Hybrid:** lock the top 60% of slots to best sellers, then run the greedy for the rest.

## Run it
**Data:** download the Instacart Market Basket dataset from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data) and put the six CSV files in a folder called `raw/`.
```bash
pip install -r requirements.txt
python src/prep.py --raw <folder with the Instacart CSVs> --out data   # ~15 s
python src/run_experiment.py --data data --out outputs                  # ~2 min
python src/make_chart.py --out outputs                                  # writes assortment_chart.html
```

## Showcase app (Streamlit)
An interactive app that explains the project in plain language: store-size slider, results,
a simulator that checks real customer orders against each strategy, and the products that change.

```bash
streamlit run app/app.py
```
It runs only from the small files in `app/data/`, so it works and deploys **without** the raw data.
To refresh those files after re-running the experiment: `python src/export_app_data.py`.

**Put it online (free):** push this repo to GitHub → go to share.streamlit.io → *Create app* →
pick the repo, branch `main`, main file `app/app.py` → *Deploy*. You get a public link to share.

## Outputs
* `assortment_chart.html` - interactive chart page
* `completion_curves.csv` - all three strategies at 500–10,000 SKUs
* `summary.json` - headline numbers
* `ranking_popularity.csv`, `ranking_basket_greedy.csv`, `assortment_hybrid_5000.csv` - the SKU lists
* `swaps_at_5000.csv` - what basket-aware adds and drops vs popularity
* `alpha_tuning.csv` - validation results per α

## Caveats
* US grocery baskets (median 8–9 items) are bigger than quick-commerce baskets, so absolute rates
  differ. The method and the direction of the gap carry over.
* No prices, margins, shelf life or per-store demand in the data. A real version would weight orders by GMV
  and run per dark store.

## Next steps
1. Substitution layer: suggest an in-stock alternative for dropped SKUs and re-measure.
2. Experiment design: A/B the hybrid assortment across matched dark stores; primary metric = order fill rate,
   guardrails = item availability, cancellations, GMV per order.
