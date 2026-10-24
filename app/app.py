"""
Dark-Store Assortment Optimiser - showcase app.

Run locally:   streamlit run app/app.py
Runs entirely from the small files in app/data (no raw data needed).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- settings --
AUTHOR = "Nirdosh Kumar"
AUTHOR_LINE = "BTech, IIT Indore"
GITHUB_URL = ""          # paste your repo link here, e.g. "https://github.com/<you>/dark-store-assortment"
DATA = Path(__file__).parent / "data"

STRATS = {
    "pop":    {"name": "Best-sellers",  "long": "Stock the most popular products", "color": "#8a8984"},
    "greedy": {"name": "Basket-aware",  "long": "Stock what completes the most orders", "color": "#2a78d6"},
    "hybrid": {"name": "Hybrid ★",      "long": "Keep top 60% best-sellers, optimise the rest", "color": "#eb6834"},
}
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#ffffff"

st.set_page_config(page_title="Dark-Store Assortment Optimiser", page_icon="🛒", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1150px;}
  h1 {font-size: 2.1rem !important; margin-bottom: .2rem !important;}
  .lede {font-size: 1.12rem; color: #3a3936; line-height: 1.6; margin: .3rem 0 1.2rem;}
  .byline {color: #6b6a66; font-size: .92rem; margin-bottom: 1.2rem;}
  .kpi {border: 1px solid #e7e6e2; border-radius: 12px; padding: 16px 18px; background: #fff; height: 100%;}
  .kpi .v {font-size: 1.75rem; font-weight: 700; color: #0b0b0b; font-variant-numeric: tabular-nums;}
  .kpi .l {font-size: .88rem; color: #52514e; margin-top: 4px; line-height: 1.4;}
  .callout {border-left: 4px solid #2a78d6; background: #f3f7fd; padding: 14px 18px; border-radius: 6px;
            margin: .6rem 0 1rem; line-height: 1.6;}
  .basket {display: flex; gap: 10px; flex-wrap: wrap; margin: .4rem 0 .8rem;}
  .chip {border: 1px solid #e7e6e2; border-radius: 999px; padding: 6px 14px; background: #fff; font-size: .95rem;}
  .chip.miss {border-color: #e34948; background: #fdf0f0; color: #9b1c1c; text-decoration: line-through;}
  .verdict {border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; border: 1px solid #e7e6e2;}
  .verdict .h {font-weight: 700;}
  .ok {background: #effaf3; border-color: #9fdcb5;}
  .bad {background: #fdf0f0; border-color: #f0b4b4;}
  .muted {color: #6b6a66; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------- data --
@st.cache_data
def load():
    curves = pd.read_csv(DATA / "completion_curves.csv")
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    tuning = pd.read_csv(DATA / "alpha_tuning.csv")
    rank_pop = pd.read_csv(DATA / "ranking_popularity.csv")
    rank_gr = pd.read_csv(DATA / "ranking_basket_greedy.csv")
    hybrid = pd.read_csv(DATA / "hybrid_assortments.csv.gz")
    orders = pd.read_csv(DATA / "sample_orders.csv.gz")
    return curves, summary, tuning, rank_pop, rank_gr, hybrid, orders


curves, summary, tuning, rank_pop, rank_gr, hybrid, orders = load()
SIZES = curves["skus"].tolist()
products = pd.concat([rank_pop, rank_gr]).drop_duplicates("product_id").set_index("product_id")


@st.cache_data
def stocked_sets(n: int) -> dict:
    return {
        "pop": set(rank_pop["product_id"].head(n)),
        "greedy": set(rank_gr["product_id"].head(n)),
        "hybrid": set(hybrid.loc[hybrid["skus"] == n, "product_id"]),
    }


@st.cache_data
def order_status(n: int) -> pd.DataFrame:
    """Per sample order: number of missing items under each strategy."""
    s = stocked_sets(n)
    df = orders[["order", "product_id"]].copy()
    for k, ids in s.items():
        df[k] = ~df["product_id"].isin(ids)
    return df.groupby("order")[list(s)].sum()


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def base_layout(fig, height=420, ytitle="", yfmt=".0%"):
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK2, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID_C, linecolor="#c9c8c2", tickfont=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID_C, tickformat=yfmt, title=ytitle, tickfont=dict(color=INK2),
                     rangemode="tozero")
    return fig


# ------------------------------------------------------------------ header --
st.title("🛒 Which products should a 10-minute delivery store stock?")
st.markdown(
    "<div class='lede'>Quick-commerce apps deliver from small <b>dark stores</b> that hold only ~5,000 products. "
    "The usual approach is to stock the best-sellers. This project shows that this <b>covers most items but fails "
    "most orders</b> – because one missing item can ruin a whole order – and builds a smarter way to pick products.</div>",
    unsafe_allow_html=True)
byline = f"Built by <b>{AUTHOR}</b> · {AUTHOR_LINE} · Data: Instacart, 3.3M real grocery orders"
if GITHUB_URL:
    byline += f" · <a href='{GITHUB_URL}' target='_blank'>Code on GitHub</a>"
st.markdown(f"<div class='byline'>{byline}</div>", unsafe_allow_html=True)

s = summary
kpis = [
    (f"{pct(s['popularity_complete'])} → {pct(s['greedy_complete'])}",
     f"orders fully in stock at {s['store_size']:,} products (best-sellers → basket-aware)"),
    (f"+{s['lift_relative'] * 100:.0f}%", "more complete orders from the same shelf space"),
    (f"~{s['popularity_skus_needed_to_match'] - s['store_size']:,}",
     "extra products the best-seller approach needs to match it"),
    (f"{s['test_orders'] // 1000}K", "real orders used to test it – orders the model never saw"),
]
for col, (v, l) in zip(st.columns(4), kpis):
    col.markdown(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>",
                 unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------- store-size input --
with st.sidebar:
    st.header("Dark store size")
    N = st.select_slider("Products the store can hold", options=SIZES, value=s["store_size"],
                         format_func=lambda x: f"{x:,}")
    st.caption("Real dark stores hold roughly 5,000–7,000 products. Every tab updates with this.")
    st.divider()
    st.markdown("**The three strategies**")
    for k, v in STRATS.items():
        st.markdown(f"<span style='color:{v['color']};font-size:1.2rem'>●</span> **{v['name']}** – {v['long']}",
                    unsafe_allow_html=True)

row = curves.set_index("skus").loc[N]
tabs = st.tabs(["1 · The problem", "2 · Results", "3 · Try a real order", "4 · What changes on the shelf",
                "5 · How it works"])

# ------------------------------------------------------------ tab 1: problem --
with tabs[0]:
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.subheader("Customers order baskets, not single items")
        st.markdown(
            "Imagine an order for **milk, bread, eggs and peanut butter**. If the store has the first three but not "
            "the peanut butter, the customer either cancels, waits, or gets a partial order. For a 10-minute app, "
            "that order is effectively lost.")
        st.markdown(
            "<div class='basket'><span class='chip'>🥛 Milk</span><span class='chip'>🍞 Bread</span>"
            "<span class='chip'>🥚 Eggs</span><span class='chip miss'>🥜 Peanut butter</span></div>",
            unsafe_allow_html=True)
        st.markdown(
            "So the right question isn't *\"which products sell the most?\"* – it's "
            "*\"which products let us fulfil the most **complete** orders?\"*")
        st.markdown(
            f"<div class='callout'>With <b>{N:,}</b> best-selling products in stock, "
            f"<b>{pct(row['pop_item_coverage'], 0)}</b> of all items customers order are available – "
            f"but only <b>{pct(row['pop_complete'], 0)}</b> of orders can be delivered complete.</div>",
            unsafe_allow_html=True)
    with c2:
        fig = go.Figure(go.Bar(
            x=["Items ordered<br>that are in stock", "Orders that are<br>fully in stock"],
            y=[row["pop_item_coverage"], row["pop_complete"]],
            marker_color=["#8a8984", "#e34948"], marker_line=dict(color="white", width=2),
            text=[pct(row["pop_item_coverage"]), pct(row["pop_complete"])], textposition="outside",
            textfont=dict(color=INK, size=15), hovertemplate="%{y:.1%}<extra></extra>", width=0.5))
        base_layout(fig, height=380, ytitle=f"Best-seller store, {N:,} products")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------------ tab 2: results --
with tabs[1]:
    st.subheader(f"Orders fully in stock at {N:,} products")
    cols = st.columns(3)
    for col, k in zip(cols, STRATS):
        v = row[f"{k}_complete"]
        delta = None if k == "pop" else f"{100 * (v - row['pop_complete']):+.1f} pp vs best-sellers"
        col.metric(STRATS[k]["name"], pct(v), delta)

    fig = go.Figure()
    for k, v in STRATS.items():
        fig.add_trace(go.Scatter(
            x=curves["skus"], y=curves[f"{k}_complete"], name=v["name"], mode="lines",
            line=dict(color=v["color"], width=2.5, dash="dot" if k == "pop" else "solid"),
            hovertemplate=f"{v['name']}: %{{y:.1%}}<extra></extra>"))
    fig.add_vline(x=N, line=dict(color="#8a8984", width=1, dash="dash"))
    base_layout(fig, height=400, ytitle="Orders fully in stock")
    fig.update_xaxes(title="Products stocked in the store", tickformat=",")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Tested on 131,209 real orders the model never saw. The smarter approaches win at every store size.")

    st.subheader("Every choice has a trade-off")
    metrics = [("Orders fully<br>in stock", "complete"), ("Orders with at most<br>1 item missing", "max1_missing"),
               ("Items ordered<br>that are in stock", "item_coverage")]
    fig2 = go.Figure()
    for k, v in STRATS.items():
        vals = [row[f"{k}_{m}"] for _, m in metrics]
        fig2.add_trace(go.Bar(
            x=[m for m, _ in metrics], y=vals, name=v["name"], marker_color=v["color"],
            marker_line=dict(color="white", width=2), text=[pct(x) for x in vals], textposition="outside",
            textfont=dict(color=INK2, size=12), hovertemplate=f"{v['name']}: %{{y:.1%}}<extra></extra>"))
    base_layout(fig2, height=400)
    fig2.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig2.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
    st.markdown(
        f"<div class='callout'><b>Recommendation: the Hybrid.</b> Pure basket-aware stocking completes the most orders "
        f"but drops some everyday staples, so fewer individual items are available "
        f"({pct(row['greedy_item_coverage'])} vs {pct(row['pop_item_coverage'])}). The Hybrid keeps the top 60% "
        f"best-sellers and optimises the rest: <b>{100 * (row['hybrid_complete'] - row['pop_complete']):+.1f} pp</b> "
        f"more complete orders while losing only "
        f"<b>{100 * (row['pop_item_coverage'] - row['hybrid_item_coverage']):.1f} pp</b> of item availability.</div>",
        unsafe_allow_html=True)

# ----------------------------------------------------- tab 3: order simulator --
with tabs[2]:
    st.subheader("Pick a real customer order and see which store could deliver it")
    status = order_status(N)
    differs = status.index[(status["pop"] > 0) & (status["hybrid"] == 0)]
    if "order_id" not in st.session_state:
        one_off = status.index[(status["pop"] == 1) & (status["hybrid"] == 0) & (status["greedy"] == 0)]
        st.session_state.order_id = int(one_off[0]) if len(one_off) else int(status.index[0])

    b1, b2, _ = st.columns([1.3, 1, 2])
    rng = np.random.default_rng()
    if b1.button("🎯 Show an order where strategy matters", width="stretch") and len(differs):
        st.session_state.order_id = int(rng.choice(differs))
    if b2.button("🎲 Random order", width="stretch"):
        st.session_state.order_id = int(rng.choice(status.index))

    oid = st.session_state.order_id
    basket = orders[orders["order"] == oid]
    sets = stocked_sets(N)
    st.markdown(f"**Order #{oid}** · {len(basket)} items · store size {N:,} products")

    vcols = st.columns(3)
    for col, k in zip(vcols, STRATS):
        missing = basket[~basket["product_id"].isin(sets[k])]
        ok = len(missing) == 0
        msg = "✅ Delivered complete" if ok else f"❌ {len(missing)} item{'s' if len(missing) > 1 else ''} missing"
        detail = "" if ok else "<div class='muted'>" + ", ".join(missing["product_name"].head(3)) + \
            ("…" if len(missing) > 3 else "") + "</div>"
        col.markdown(f"<div class='verdict {'ok' if ok else 'bad'}'>"
                     f"<div class='muted'>{STRATS[k]['name']}</div><div class='h'>{msg}</div>{detail}</div>",
                     unsafe_allow_html=True)

    table = basket[["product_name", "aisle"]].copy()
    for k in STRATS:
        table[STRATS[k]["name"]] = np.where(basket["product_id"].isin(sets[k]), "✅", "❌")
    table.columns = ["Product", "Aisle"] + [STRATS[k]["name"] for k in STRATS]
    st.dataframe(table, hide_index=True, width="stretch")

    rates = (status == 0).mean()
    st.caption(f"Across all {len(status):,} sample orders at this store size, complete orders: "
               + " · ".join(f"{STRATS[k]['name']} {pct(rates[k])}" for k in STRATS)
               + ". (Random sample of 3,000 test orders with 2–20 items, so rates differ slightly from the headline.)")

# ----------------------------------------------------- tab 4: shelf changes --
with tabs[3]:
    st.subheader(f"What the Hybrid changes vs a best-seller store ({N:,} products)")
    sets = stocked_sets(N)
    added = list(sets["hybrid"] - sets["pop"])
    dropped = list(sets["pop"] - sets["hybrid"])
    pop_rank = rank_pop.set_index("product_id")["rank"]
    st.markdown(
        f"The Hybrid swaps **{len(added):,}** of the {N:,} products. It drops lower-ranked best-sellers whose orders "
        "are usually missing other items anyway, and adds products that are often the **last missing item** in an "
        "otherwise fully stocked order.")
    c1, c2 = st.columns(2)

    def swap_table(ids, ascending):
        d = products.loc[ids, ["product_name", "aisle"]].copy()
        d["Best-seller rank"] = pop_rank.reindex(ids).fillna(np.inf).values
        d = d.sort_values("Best-seller rank", ascending=ascending).head(12)
        d["Best-seller rank"] = d["Best-seller rank"].map(lambda r: "outside top 10K" if np.isinf(r) else f"#{int(r):,}")
        d.columns = ["Product", "Aisle", "Best-seller rank"]
        d = d[["Product", "Best-seller rank", "Aisle"]]
        return d

    with c1:
        st.markdown("**➕ Added by the Hybrid**")
        st.dataframe(swap_table(added, True), hide_index=True, width="stretch")
    with c2:
        st.markdown("**➖ Dropped from the best-seller list**")
        st.dataframe(swap_table(dropped, True), hide_index=True, width="stretch")

    dep_of = products["department"].replace({"missing": "uncategorised"})
    dep = pd.DataFrame({
        "Added": dep_of.loc[added].value_counts(),
        "Dropped": dep_of.loc[dropped].value_counts(),
    }).fillna(0).astype(int)
    dep["Net"] = dep["Added"] - dep["Dropped"]
    dep = dep.sort_values("Net")
    fig = go.Figure(go.Bar(
        y=dep.index, x=dep["Net"], orientation="h",
        marker_color=np.where(dep["Net"] >= 0, "#2a78d6", "#8a8984"), marker_line=dict(color="white", width=1),
        hovertemplate="%{y}: %{x:+d} products<extra></extra>"))
    base_layout(fig, height=max(320, 22 * len(dep)), yfmt="")
    fig.update_xaxes(title="Net products added (+) or removed (−) by department")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------- tab 5: how it works --
with tabs[4]:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Method")
        st.markdown(f"""
1. **Data** – Instacart Market Basket: 3.3M real grocery orders, 33.8M order lines, 49.7K products.
2. **Fair testing** – products are chosen using 1M past orders, the algorithm is tuned on 200K other
   orders, and results are reported on each customer's **latest** order (131K orders the model never saw).
3. **Best-sellers baseline** – rank products by how many orders they appear in.
4. **Basket-aware algorithm** – add products one at a time. Every order "votes" for the items it is still
   missing, and an order that is only 1–2 items away from complete gets a much stronger vote. Stock the product
   with the most votes, update, repeat. (Vote strength tuned on validation: α = {s['best_alpha']}.)
5. **Hybrid** – reserve the top 60% of shelf space for best-sellers, then run the basket-aware algorithm on the rest.
""")
        st.subheader("How I'd test it in the real world")
        st.markdown("""
- **A/B test** the Hybrid assortment in a set of dark stores vs matched control stores.
- **Primary metric:** share of orders delivered complete (fill rate).
- **Guardrails:** item availability, cancellations, substitutions, order value.
""")
    with c2:
        st.subheader("Limitations")
        st.markdown("""
- US grocery baskets (median 8–9 items) are bigger than quick-commerce baskets, so exact percentages
  will differ – the method and the direction of the result carry over.
- The data has no prices, margins or shelf life; a real version would weight orders by value and run per store.
- Substitutes (e.g. another brand of peanut butter) aren't modelled yet – the natural next step.
""")
        st.subheader("Tuning on validation orders")
        t = tuning.rename(columns={"alpha": "α (vote strength)", "mean_completion": "Avg. complete orders",
                                   f"completion_at_{s['store_size']}": f"At {s['store_size']:,} products"})
        st.dataframe(t.style.format({t.columns[0]: "{:.2f}", **{c: "{:.1%}" for c in t.columns[1:]}}),
                     hide_index=True,
                     width="stretch")
        st.caption("α = 1 behaves like counting popularity; smaller α focuses on nearly-complete orders.")

st.divider()
st.caption(f"{AUTHOR} · Dark-Store Assortment Optimiser · Python, Pandas, NumPy, SciPy, Plotly, Streamlit")
