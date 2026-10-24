"""
Step 4 - Chart: SKUs stocked vs share of orders that can be filled completely.

Usage:  python src/make_chart.py --out outputs
Writes outputs/assortment_chart.html (interactive, opens offline).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e2", "#fcfcfb"
SERIES = {  # baseline recedes in gray; the two new strategies take categorical slots 1-2
    "pop":    ("Popularity (stock the best sellers)", "#8a8984", "dot"),
    "greedy": ("Basket-aware (complete the most orders)", "#2a78d6", "solid"),
    "hybrid": ("Hybrid (lock top 60% sellers, optimise rest)", "#eb6834", "solid"),
}
FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"


def layout(fig, title, subtitle, ytitle, height=470):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{INK2}'>{subtitle}</span>",
                   x=0, xanchor="left", font=dict(size=18, color=INK)),
        font=dict(family=FONT, color=INK2, size=13),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        margin=dict(l=60, r=30, t=90, b=60),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=MUTED, zeroline=False, tickfont=dict(color=INK2))
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, tickformat=".0%",
                     tickfont=dict(color=INK2), rangemode="tozero")
    return fig


def main(out: Path) -> None:
    c = pd.read_csv(out / "completion_curves.csv")
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    n = s["store_size"]

    # ---- chart 1: completion curve -----------------------------------------
    f1 = go.Figure()
    for key, (name, color, dash) in SERIES.items():
        f1.add_trace(go.Scatter(
            x=c["skus"], y=c[f"{key}_complete"], name=name, mode="lines",
            line=dict(color=color, width=2.5 if key != "pop" else 2, dash=dash),
            hovertemplate=f"{name.split(' (')[0]}: %{{y:.1%}}<extra></extra>"))
    f1.add_vline(x=n, line=dict(color=MUTED, width=1, dash="dash"))
    f1.add_annotation(x=n, y=1.02, yref="paper", text=f"Typical dark store ({n:,} SKUs)",
                      showarrow=False, font=dict(size=12, color=INK2), xanchor="left", xshift=4)
    for key, ax in [("greedy", -60), ("pop", 40)]:
        v = s[f"{'popularity' if key == 'pop' else 'greedy'}_complete"]
        f1.add_annotation(x=n, y=v, text=f"<b>{v:.1%}</b>", ax=ax, ay=-25 if key == "greedy" else 25,
                          arrowcolor=MUTED, arrowwidth=1, font=dict(color=INK, size=13))
    layout(f1, "Basket-aware stocking completes more orders at every store size",
           f"Share of held-out orders ({s['test_orders']:,} users' latest orders) where every item is in stock",
           "Orders fully in stock")
    f1.update_xaxes(title="SKUs stocked in the dark store", tickformat=",", range=[400, 10100])
    f1.update_layout(hovermode="x unified")

    # ---- chart 2: the trade-off at store size -------------------------------
    row = c.set_index("skus").loc[n]
    metrics = [("Orders fully<br>in stock", "complete"),
               ("Orders with ≤1<br>item missing", "max1_missing"),
               ("Items ordered<br>that are in stock", "item_coverage")]
    f2 = go.Figure()
    for key, (name, color, _) in SERIES.items():
        vals = [row[f"{key}_{m}"] for _, m in metrics]
        f2.add_trace(go.Bar(
            x=[m for m, _ in metrics], y=vals, name=name.split(" (")[0], marker_color=color,
            marker_line=dict(color=SURFACE, width=2),
            text=[f"{v:.1%}" for v in vals], textposition="outside",
            textfont=dict(color=INK2, size=12),
            hovertemplate=f"{name}<br>%{{y:.1%}}<extra></extra>"))
    layout(f2, f"The trade-off at {n:,} SKUs",
           "Basket-aware wins on complete orders but gives up some popular items; the hybrid keeps most of both",
           "Share", height=430)
    f2.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    f2.update_yaxes(range=[0, 1])
    f2.update_xaxes(tickangle=0)

    # ---- page ---------------------------------------------------------------
    tiles = [
        (f"{s['popularity_complete']:.1%} → {s['greedy_complete']:.1%}",
         f"orders fully in stock at {n:,} SKUs (popularity → basket-aware)"),
        (f"+{s['lift_relative']:.0%}", "more complete orders from the same shelf space"),
        (f"~{s['popularity_skus_needed_to_match'] - n:,}",
         "extra SKUs popularity needs to match it"),
        (f"+{s['hybrid_lift_pp']:.1f} pp",
         f"hybrid gain, losing only {100*(s['popularity_item_coverage']-s['hybrid_item_coverage']):.1f} pp item coverage"),
    ]
    tile_html = "".join(
        f"<div class='tile'><div class='big'>{a}</div><div class='lbl'>{b}</div></div>" for a, b in tiles)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dark-Store Assortment</title>
<style>
 body{{margin:0;background:{SURFACE};color:{INK};font-family:{FONT}}}
 main{{max-width:1000px;margin:0 auto;padding:28px 16px 48px}}
 h1{{font-size:24px;margin:0 0 6px}} p.sub{{color:{INK2};margin:0 0 22px;line-height:1.5}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:22px}}
 .tile{{border:1px solid {GRID};border-radius:10px;padding:14px 16px;background:white}}
 .big{{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}}
 .lbl{{color:{INK2};font-size:13px;margin-top:4px;line-height:1.4}}
 .chart{{border:1px solid {GRID};border-radius:10px;background:{SURFACE};margin-bottom:18px;padding:6px}}
 .note{{color:{INK2};font-size:13px;line-height:1.55}}
</style></head><body><main>
<h1>Which SKUs should a dark store stock?</h1>
<p class="sub">Stocking the best sellers maximises items covered, not orders completed. In quick commerce one missing
item can lose the whole order. SKUs chosen on 1M past orders; results measured on {s['test_orders']:,} later orders the
model never saw (Instacart Market Basket data).</p>
<div class="tiles">{tile_html}</div>
<div class="chart">{f1.to_html(full_html=False, include_plotlyjs=True, config={"displayModeBar": False, "responsive": True})}</div>
<div class="chart">{f2.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})}</div>
<p class="note"><b>How basket-aware selection works:</b> products are added one at a time; each step stocks the product that
moves the most orders closer to complete, weighting orders that are only 1–2 items away. Weight decay tuned on a
separate 200k-order validation set (α = {s['best_alpha']}). <b>Caveat:</b> US grocery baskets (median 8–9 items) are larger
than Blinkit's, so absolute rates differ; the method and the direction of the gap carry over.</p>
</main></body></html>"""
    (out / "assortment_chart.html").write_text(html, encoding="utf-8")
    print("wrote", out / "assortment_chart.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    main(ap.parse_args().out)
