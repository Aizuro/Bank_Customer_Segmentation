"""Dashboard Segmentasi Nasabah (versi minimalis).

Jalankan:  streamlit run app.py
Prasyarat: jalankan Customer_Segmentation_Improved.ipynb sampai selesai agar folder `artifacts/` terbentuk.
"""
import base64
import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Segmentasi Nasabah", page_icon="📊", layout="wide")

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
REQUIRED_FILES = [
    "customers_with_segment.csv", "segment_profile.csv", "segment_profile_zscore.csv",
    "transactions_with_segment.csv", "metadata.json", "segmentation_pipeline.joblib",
    "segment_classifier.joblib",
]

# ---------------------------------------------------------------------------------------------
# Kamus istilah: nama ramah, kalimat awam, format angka
# ---------------------------------------------------------------------------------------------
MONTHS_ID = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
OCCUPATION_ID = {"Doctor": "Dokter", "Engineer": "Insinyur", "Retired": "Pensiunan", "Student": "Pelajar"}
PALETTE = ["#0F9D8A", "#E8A33D", "#6366F1", "#E5586E", "#38A3D8", "#84A93C"]
MIN_Z = 0.3          # ciri dengan |z| di bawah ini dianggap tidak menonjol
DOT_HINT = "#7B8190"  # warna teks di dalam gambar SVG (terbaca di tema terang maupun gelap)


def idn(v, dec=0):
    """Format angka gaya Indonesia: 1.234,5"""
    s = f"{v:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def f_count(v): return idn(v, 1)
def f_money(v): return idn(v, 0)
def f_pct(v): return idn(v * 100, 0) + "%"
def f_sec(v): return idn(v, 0) + " dtk"
def f_days(v): return idn(v, 0) + " hari"


# fitur -> (nama ramah, kalimat jika tinggi, kalimat jika rendah, pemformat angka, faktor skala sumbu grafik)
FEATURES = {
    "n_txn": ("Jumlah transaksi", "Lebih sering bertransaksi", "Lebih jarang bertransaksi", f_count, 1),
    "avg_amount": ("Rata-rata nominal transaksi", "Nominal transaksi lebih besar", "Nominal transaksi lebih kecil", f_money, 1),
    "avg_balance": ("Rata-rata saldo", "Saldo lebih besar", "Saldo lebih kecil", f_money, 1),
    "avg_amount_to_balance": ("Nominal dibanding saldo", "Transaksi besar dibanding saldonya", "Transaksi kecil dibanding saldonya", f_pct, 100),
    "avg_duration": ("Durasi transaksi", "Transaksi lebih lama", "Transaksi lebih singkat", f_sec, 1),
    "multi_login_rate": ("Login lebih dari 1x", "Lebih sering mengulang login", "Jarang mengulang login", f_pct, 100),
    "online_share": ("Kanal Online", "Lebih sering lewat Online", "Jarang lewat Online", f_pct, 100),
    "atm_share": ("Kanal ATM", "Lebih sering lewat ATM", "Jarang lewat ATM", f_pct, 100),
    "debit_share": ("Transaksi Debit", "Lebih banyak transaksi debit", "Lebih banyak transaksi kredit", f_pct, 100),
    "n_locations": ("Jumlah lokasi", "Memakai lebih banyak lokasi", "Memakai lebih sedikit lokasi", f_count, 1),
    "avg_gap_days": ("Jeda antar transaksi", "Jeda antar transaksi lebih lama", "Jeda antar transaksi lebih singkat", f_days, 1),
}
INT_FEATURES = {"n_txn", "n_locations"}
SHARE_FEATURES = {"multi_login_rate", "online_share", "atm_share", "debit_share"}


def fname(f): return FEATURES.get(f, (f,))[0]
def esc(x): return html.escape(str(x))


# ---------------------------------------------------------------------------------------------
# Gaya global & helper HTML
# ---------------------------------------------------------------------------------------------
CSS = """
<style>
.block-container{max-width:1080px;padding-top:2.2rem;padding-bottom:4rem;}
footer{visibility:hidden;}
.app-title{font-size:1.9rem;font-weight:700;letter-spacing:-.01em;line-height:1.2;margin:0;}
.app-sub{opacity:.62;margin:.2rem 0 1.2rem 0;font-size:.98rem;}
.kpis{display:flex;gap:2.6rem;flex-wrap:wrap;align-items:flex-end;margin:.2rem 0 1.1rem 0;}
.kpi .v{font-size:1.75rem;font-weight:650;line-height:1.1;}
.kpi .l{font-size:.8rem;opacity:.6;margin-top:.2rem;}
.chip{display:inline-block;padding:.14rem .65rem;border-radius:999px;font-size:.82rem;font-weight:600;}
.chip.weak{background:rgba(232,163,61,.17);color:#C07A0C;}
.chip.mid{background:rgba(56,163,216,.16);color:#1E80B3;}
.chip.strong{background:rgba(15,157,138,.16);color:#0B8272;}
.sharebar{display:flex;gap:3px;height:12px;border-radius:999px;overflow:hidden;margin:.4rem 0 .6rem 0;}
.sharebar i{display:block;height:100%;}
.legend{display:flex;gap:1.4rem;flex-wrap:wrap;font-size:.86rem;margin-bottom:1.3rem;}
.legend span.d{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:.4rem;}
.legend .m{opacity:.6;margin-left:.3rem;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin-bottom:.6rem;}
.card{border:1px solid rgba(128,128,128,.24);border-radius:14px;padding:18px 20px 16px 20px;}
.card .h{font-weight:650;font-size:1.05rem;display:flex;align-items:center;gap:.5rem;}
.card .dot{display:inline-block;width:10px;height:10px;border-radius:50%;}
.card .sub{font-size:.82rem;opacity:.6;margin:.15rem 0 1rem 0;}
.card .tag{font-size:.92rem;margin-bottom:.9rem;line-height:1.35;}
.trait{margin:.62rem 0;}
.trait .t{font-size:.86rem;font-weight:600;}
.trait .v{font-size:.8rem;opacity:.62;margin:.05rem 0 .3rem 0;}
.meter{height:5px;border-radius:999px;background:rgba(128,128,128,.18);overflow:hidden;}
.meter i{display:block;height:100%;border-radius:999px;}
.card .note{font-size:.84rem;opacity:.75;margin-top:.9rem;padding-top:.7rem;border-top:1px dashed rgba(128,128,128,.3);}
.card .none{font-size:.85rem;opacity:.6;}
.hint{font-size:.84rem;opacity:.62;line-height:1.5;}
.ptable{width:100%;border-collapse:separate;border-spacing:3px;font-size:.88rem;}
.ptable th{font-weight:600;text-align:center;padding:6px 4px;font-size:.84rem;}
.ptable td{text-align:center;padding:9px 6px;border-radius:7px;}
.ptable td.f{text-align:left;opacity:.85;padding-left:2px;}
.months{display:flex;align-items:flex-end;gap:10px;height:190px;padding-top:6px;}
.mcol{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;min-width:0;}
.mstack{width:100%;max-width:46px;display:flex;flex-direction:column-reverse;border-radius:5px;overflow:hidden;gap:1px;}
.mstack i{display:block;width:100%;}
.mtot{font-size:.72rem;opacity:.55;margin-bottom:3px;}
.mlab{font-size:.74rem;opacity:.6;margin-top:6px;white-space:nowrap;}
.result{border:1px solid rgba(128,128,128,.24);border-left-width:6px;border-radius:14px;padding:18px 22px;margin:.4rem 0 1rem 0;}
.result .l{font-size:.8rem;opacity:.6;}
.result .n{font-size:1.5rem;font-weight:700;margin:.1rem 0 .2rem 0;}
.result .d{font-size:.95rem;margin-bottom:.6rem;}
.about td{padding:.45rem .8rem .45rem 0;vertical-align:top;font-size:.9rem;}
.about td.k{font-weight:600;white-space:nowrap;}
.about td.e{opacity:.65;}
</style>
"""


def put(s):
    """Kirim HTML ke Streamlit. Baris dirapatkan agar parser Markdown tidak memecah blok HTML."""
    st.markdown("".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def seg_colors(segment_ids):
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sorted(segment_ids))}


# ---------------------------------------------------------------------------------------------
# Memuat artefak
# ---------------------------------------------------------------------------------------------
@st.cache_data
def load_tables():
    cust = pd.read_csv(ARTIFACT_DIR / "customers_with_segment.csv")
    profile = pd.read_csv(ARTIFACT_DIR / "segment_profile.csv", index_col=0)
    zprof = pd.read_csv(ARTIFACT_DIR / "segment_profile_zscore.csv", index_col=0)
    tx = pd.read_csv(ARTIFACT_DIR / "transactions_with_segment.csv", parse_dates=["TransactionDate"])
    meta = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    notes_path = ARTIFACT_DIR / "segment_notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    return cust, profile, zprof, tx, meta, notes


@st.cache_resource
def load_models():
    seg_pipe = joblib.load(ARTIFACT_DIR / "segmentation_pipeline.joblib")
    clf = joblib.load(ARTIFACT_DIR / "segment_classifier.joblib")
    return seg_pipe, clf


# ---------------------------------------------------------------------------------------------
# Logika tampilan (fungsi murni agar mudah diuji)
# ---------------------------------------------------------------------------------------------
def validity_level(meta):
    """Aturan sama dengan tabel interpretasi di notebook (bagian 4)."""
    sil, null, ari = meta["silhouette"], meta["silhouette_null_mean"], meta["stability_ari"]
    if sil > 0.5 and (sil - null) > 0.2 and ari > 0.8:
        return "strong", "Kuat", "Segmen terpisah jelas dan stabil. Kelompok ini benar-benar ada di data."
    if sil >= 0.25 and ari >= 0.6:
        return "mid", "Sedang", "Ada pola, tetapi batas antar segmen tumpang tindih. Pakai sebagai pengelompokan praktis."
    return "weak", "Lemah", ("Nasabah di data ini tidak membentuk kelompok yang terpisah jelas. Segmen tetap berguna untuk "
                             "membaca perbedaan perilaku, tetapi bukan kelompok alami.")


def top_traits(feature_values, feature_means, z_row, features, n=3):
    """Ambil ciri paling menonjol (|z| >= MIN_Z) beserta kalimat awam dan angkanya."""
    z = z_row[[f for f in features if f in z_row.index]]
    out = []
    for f in z.abs().sort_values(ascending=False).index:
        if abs(z[f]) < MIN_Z or f not in FEATURES:
            continue
        _, hi, lo, fmt, _ = FEATURES[f]
        out.append({"feature": f, "z": float(z[f]), "phrase": hi if z[f] > 0 else lo,
                    "value": fmt(feature_values[f]), "avg": fmt(feature_means[f])})
        if len(out) == n:
            break
    return out


def share_bar_html(counts, names, colors):
    total = counts.sum()
    bar = "".join(f'<i style="width:{c / total * 100:.2f}%;background:{colors[s]}"></i>' for s, c in counts.items())
    leg = "".join(
        f'<div><span class="d" style="background:{colors[s]}"></span>{esc(names[s])}'
        f'<span class="m">{c / total * 100:.0f}%</span></div>' for s, c in counts.items())
    return f'<div class="sharebar">{bar}</div><div class="legend">{leg}</div>'


def segment_cards_html(profile, zprof, cust, features, names, colors, notes):
    means = cust[features].mean()
    total = len(cust)
    cards = []
    for seg_id in sorted(profile.index):
        n = int((cust["segment"] == seg_id).sum())
        traits = top_traits(profile.loc[seg_id], means, zprof.loc[seg_id], features)
        rows = "".join(
            f'<div class="trait"><div class="t">{esc(t["phrase"])}</div>'
            f'<div class="v">{esc(t["value"])} <span>(rata-rata {esc(t["avg"])})</span></div>'
            f'<div class="meter"><i style="width:{min(abs(t["z"]) / 1.5, 1) * 100:.0f}%;background:{colors[seg_id]}"></i></div></div>'
            for t in traits) or '<div class="none">Tidak ada ciri yang menonjol: perilakunya mendekati rata-rata nasabah.</div>'
        note = notes.get(names[seg_id])
        note_html = f'<div class="note">{esc(note)}</div>' if note else ""
        cards.append(
            f'<div class="card" style="border-top:3px solid {colors[seg_id]}">'
            f'<div class="h"><span class="dot" style="background:{colors[seg_id]}"></span>{esc(names[seg_id])}</div>'
            f'<div class="sub">{idn(n)} nasabah · {n / total * 100:.0f}%</div>'
            f'{rows}{note_html}</div>')
    return f'<div class="cards">{"".join(cards)}</div>'


def profile_table_html(profile, zprof, features, names, colors):
    feats = [f for f in features if f in FEATURES]
    head = "".join(f'<th><span style="color:{colors[s]}">●</span> {esc(names[s])}</th>' for s in sorted(profile.index))
    body = []
    for f in feats:
        fmt = FEATURES[f][3]
        cells = []
        for s in sorted(profile.index):
            z = float(zprof.loc[s, f])
            alpha = min(abs(z) / 1.5, 1) * 0.34
            c = colors[s].lstrip("#")
            rgb = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
            arrow = "▲" if z > 0.15 else ("▼" if z < -0.15 else "")
            cells.append(f'<td style="background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f})">{esc(fmt(profile.loc[s, f]))} '
                         f'<span style="opacity:.5;font-size:.7rem">{arrow}</span></td>')
        body.append(f'<tr><td class="f">{esc(fname(f))}</td>{"".join(cells)}</tr>')
    return f'<table class="ptable"><tr><th></th>{head}</tr>{"".join(body)}</table>'


def monthly_counts(tx):
    """Hitung transaksi per bulan per segmen. Bulan terakhir yang belum lengkap dibuang agar tidak tampak 'anjlok'."""
    t = tx.dropna(subset=["TransactionDate"]).copy()
    last = t["TransactionDate"].max()
    month_end = last + pd.offsets.MonthEnd(0)
    dropped = None
    if (month_end - last).days > 3:
        dropped = last.to_period("M")
        t = t[t["TransactionDate"].dt.to_period("M") != dropped]
    t["month"] = t["TransactionDate"].dt.to_period("M")
    pivot = t.pivot_table(index="month", columns="segment_name", values="TransactionAmount", aggfunc="count", fill_value=0)
    return pivot, dropped


def months_html(pivot, order_names, name_color):
    maxtot = float(pivot.sum(axis=1).max()) or 1.0
    bar_h = 150.0
    cols = []
    for period, row in pivot.iterrows():
        tot = float(row.sum())
        segs = "".join(
            f'<i title="{esc(n)}: {int(row[n])}" style="height:{row[n] / maxtot * bar_h:.1f}px;background:{name_color[n]}"></i>'
            for n in order_names if n in row.index and row[n] > 0)
        lab = MONTHS_ID[period.month - 1] + (f" {str(period.year)[2:]}" if period.month == 1 or period == pivot.index[0] else "")
        cols.append(f'<div class="mcol"><div class="mtot">{int(tot)}</div><div class="mstack">{segs}</div><div class="mlab">{lab}</div></div>')
    return f'<div class="months">{"".join(cols)}</div>'


def nice_ticks(lo, hi, n=5, min_step=None):
    span = max(hi - lo, 1e-9)
    step = span / n
    mag = 10 ** np.floor(np.log10(step))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= step:
            step = m * mag
            break
    if min_step and step < min_step:
        step = min_step
    if min_step == 1 and span <= 12:
        step = 1.0   # bilangan bulat dengan rentang kecil: beri label setiap angka
    start = np.ceil(lo / step - 1e-9) * step
    ticks = np.arange(start, hi + step * 0.001, step)
    ticks = np.where(np.abs(ticks) < step * 1e-9, 0.0, ticks) + 0.0   # hindari label "-0"
    return ticks, step


def tick_label(v, step):
    if step >= 1:
        return idn(v, 0)
    return idn(v, 1 if step >= 0.1 else 2)


def scatter_img(view, xf, yf, colors, names):
    """Scatter ringan sebagai gambar SVG (tanpa matplotlib)."""
    W, H, L, R, T, B = 760, 380, 58, 16, 14, 46
    sx, sy = FEATURES[xf][4], FEATURES[yf][4]
    x, y = view[xf].to_numpy(float) * sx, view[yf].to_numpy(float) * sy
    rng = np.random.default_rng(0)
    if xf in INT_FEATURES:
        x = x + rng.uniform(-0.22, 0.22, len(x))
    if yf in INT_FEATURES:
        y = y + rng.uniform(-0.22, 0.22, len(y))
    xlo, xhi = np.percentile(x, [1, 99]); ylo, yhi = np.percentile(y, [1, 99])
    px, py = (xhi - xlo) * 0.06 or 1, (yhi - ylo) * 0.06 or 1
    xlo, xhi, ylo, yhi = xlo - px, xhi + px, ylo - py, yhi + py
    if (view[xf] >= 0).all():
        xlo = max(xlo, 0.0)   # nilai tidak mungkin negatif
    if (view[yf] >= 0).all():
        ylo = max(ylo, 0.0)
    X = lambda v: L + (v - xlo) / (xhi - xlo) * (W - L - R)
    Y = lambda v: H - B - (v - ylo) / (yhi - ylo) * (H - T - B)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">']
    xt, xs = nice_ticks(xlo, xhi, min_step=1 if xf in INT_FEATURES else None)
    yt, ys = nice_ticks(ylo, yhi, min_step=1 if yf in INT_FEATURES else None)
    for v in yt:
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#8A90A0" stroke-opacity=".22"/>'
                     f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{DOT_HINT}">{tick_label(v, ys)}</text>')
    for v in xt:
        parts.append(f'<text x="{X(v):.1f}" y="{H - B + 18}" text-anchor="middle" fill="{DOT_HINT}">{tick_label(v, xs)}</text>')
    seg_col = view["segment"].to_numpy()
    inside = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    for xi, yi, s in zip(x[inside], y[inside], seg_col[inside]):
        parts.append(f'<circle cx="{X(xi):.1f}" cy="{Y(yi):.1f}" r="3.6" fill="{colors[s]}" fill-opacity=".62"/>')
    unit = lambda f: " (%)" if FEATURES[f][4] == 100 else ""
    parts.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" text-anchor="middle" fill="{DOT_HINT}">{esc(fname(xf))}{unit(xf)}</text>')
    parts.append(f'<text x="14" y="{(T + H - B) / 2:.0f}" text-anchor="middle" fill="{DOT_HINT}" transform="rotate(-90 14 {(T + H - B) / 2:.0f})">{esc(fname(yf))}{unit(yf)}</text>')
    parts.append("</svg>")
    b64 = base64.b64encode("".join(parts).encode("utf-8")).decode("ascii")
    return f'<img style="width:100%;max-width:760px" alt="Sebaran nasabah" src="data:image/svg+xml;base64,{b64}"/>'


# ---------------------------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------------------------
def render_header(cust, tx, meta):
    level, label, _ = validity_level(meta)
    start, end = tx["TransactionDate"].min(), tx["TransactionDate"].max()
    period = f"{MONTHS_ID[start.month - 1]} {start.year} – {MONTHS_ID[end.month - 1]} {end.year}"
    put(f"""
    <div class="app-title">Segmentasi Nasabah</div>
    <div class="app-sub">Pengelompokan nasabah bank berdasarkan perilaku bertransaksi · data {esc(period)}</div>
    <div class="kpis">
    <div class="kpi"><div class="v">{idn(meta['n_customers'])}</div><div class="l">Nasabah</div></div>
    <div class="kpi"><div class="v">{meta['k']}</div><div class="l">Segmen</div></div>
    <div class="kpi"><div class="v"><span class="chip {level}">{label}</span></div><div class="l">Kekuatan pengelompokan</div></div>
    </div>
    """)


def render_overview(cust, profile, zprof, tx, meta, notes, names, colors):
    features = meta["cluster_features"]
    _, label, explain = validity_level(meta)

    with st.expander(f"Kekuatan pengelompokan: {label}. Apa artinya?"):
        put(f'<div class="hint">{esc(explain)}<br><br>Detail angkanya ada di tab <b>Tentang</b>.</div>')

    counts = cust.groupby("segment").size()
    put(share_bar_html(counts, names, colors))
    put(segment_cards_html(profile, zprof, cust, features, names, colors, notes))

    st.write("")
    with st.expander("Lihat perbandingan semua ciri"):
        put('<div class="hint" style="margin-bottom:.6rem">Warna lebih pekat = lebih jauh dari rata-rata nasabah. '
            '▲ di atas rata-rata, ▼ di bawah rata-rata.</div>')
        put(profile_table_html(profile, zprof, features, names, colors))

    pivot, dropped = monthly_counts(tx)
    with st.expander("Lihat aktivitas transaksi per bulan"):
        name_color = {names[s]: colors[s] for s in names}
        put(months_html(pivot, [names[s] for s in sorted(names)], name_color))
        cap = ("Jumlah transaksi tiap bulan, dibagi menurut segmen nasabahnya. Segmen ditetapkan dari seluruh periode data, "
               "jadi ini bukan pergeseran segmen.")
        if dropped is not None:
            cap += f" {MONTHS_ID[dropped.month - 1]} {dropped.year} tidak ditampilkan karena datanya belum lengkap."
        put(f'<div class="hint" style="margin-top:.7rem">{esc(cap)}</div>')


def render_explorer(cust, meta, names, colors):
    features = meta["cluster_features"]
    c1, c2, c3 = st.columns([2, 2, 1.6])
    seg_options = [names[s] for s in sorted(names)]
    chosen_segs = c1.multiselect("Segmen", seg_options, default=seg_options)
    occ_labels = {o: OCCUPATION_ID.get(o, o) for o in sorted(cust["occupation"].dropna().unique())}
    chosen_occ = c2.multiselect("Pekerjaan", list(occ_labels.values()), default=list(occ_labels.values()))
    age_min, age_max = int(cust["age"].min()), int(cust["age"].max())
    age_range = c3.slider("Usia", age_min, age_max, (age_min, age_max))

    inv_occ = {v: k for k, v in occ_labels.items()}
    mask = (cust["segment_name"].isin(chosen_segs) & cust["occupation"].isin([inv_occ[o] for o in chosen_occ])
            & cust["age"].between(age_range[0], age_range[1]))
    view = cust[mask]
    st.caption(f"Menampilkan {idn(len(view))} dari {idn(len(cust))} nasabah")
    if view.empty:
        st.info("Tidak ada nasabah yang sesuai filter.")
        return

    show = pd.DataFrame({
        "Akun": view["AccountID"], "Segmen": view["segment_name"], "Usia": view["age"].round(0).astype(int),
        "Pekerjaan": view["occupation"].map(lambda o: OCCUPATION_ID.get(o, o)),
        "Transaksi": view["n_txn"], "Rata-rata nominal": view["avg_amount"].round(0),
        "Rata-rata saldo": view["avg_balance"].round(0),
    })
    cfg = {"Transaksi": st.column_config.NumberColumn("Transaksi", format="%d"),
           "Rata-rata nominal": st.column_config.NumberColumn("Rata-rata nominal", format="%d"),
           "Rata-rata saldo": st.column_config.NumberColumn("Rata-rata saldo", format="%d")}
    for col, src in [("Online (%)", "online_share"), ("ATM (%)", "atm_share"), ("Cabang (%)", "branch_share")]:
        if src in view.columns:
            show[col] = (view[src] * 100).round(0)
            cfg[col] = st.column_config.NumberColumn(col, format="%d")
    st.dataframe(show, hide_index=True, height=340, column_config=cfg)
    st.download_button("Unduh CSV", view.to_csv(index=False).encode("utf-8"), file_name="nasabah.csv", mime="text/csv")

    with st.expander("Bandingkan dua ciri dalam grafik"):
        a, b = st.columns(2)
        opts = [f for f in features if f in FEATURES]
        x_feat = a.selectbox("Sumbu mendatar", opts, index=0, format_func=fname)
        y_feat = b.selectbox("Sumbu tegak", opts, index=min(1, len(opts) - 1), format_func=fname)
        put(scatter_img(view, x_feat, y_feat, colors, names))
        leg = "".join(f'<div><span class="d" style="background:{colors[s]}"></span>{esc(names[s])}</div>' for s in sorted(names))
        put(f'<div class="legend" style="margin-top:.4rem">{leg}</div>')
        put('<div class="hint">Titik yang sangat jauh (di luar 1–99% data) tidak ditampilkan agar grafik mudah dibaca.</div>')


def render_simulator(cust, profile, zprof, meta, seg_pipe, clf, names, colors):
    features = meta["cluster_features"]
    put('<div class="hint" style="margin-bottom:.8rem">Isi ringkasan perilaku seorang nasabah, lalu lihat ia paling mirip segmen yang mana. '
        'Nilai awal adalah median nasabah.</div>')

    groups = [
        ("Aktivitas", ["n_txn", "n_locations", "avg_gap_days"]),
        ("Nominal", ["avg_amount", "avg_balance", "avg_duration"]),
        ("Kebiasaan", ["online_share", "atm_share", "debit_share", "multi_login_rate"]),
    ]
    values = {}
    with st.form("simulator", border=False) if _supports_form_border() else st.form("simulator"):
        cols = st.columns(3)
        for col, (title, fs) in zip(cols, groups):
            with col:
                st.markdown(f"**{title}**")
                for f in [x for x in fs if x in features]:
                    series = cust[f]
                    med, lo, hi = float(series.median()), float(series.min()), float(series.max())
                    help_txt = f"Rentang di data: {idn(lo * FEATURES[f][4], 1)} sampai {idn(hi * FEATURES[f][4], 1)}" + (" (%)" if FEATURES[f][4] == 100 else "")
                    if f in SHARE_FEATURES:
                        values[f] = st.slider(fname(f), 0.0, 1.0, float(round(med, 2)), 0.05, help=help_txt)
                    elif f in INT_FEATURES:
                        values[f] = st.number_input(fname(f), min_value=0, value=int(round(med)), step=1, help=help_txt)
                    else:
                        step = float(round(max(med / 20, 0.01), 2))
                        values[f] = st.number_input(fname(f), min_value=0.0, value=float(round(med, 2)), step=step, help=help_txt)
        submitted = st.form_submit_button("Cari segmen")

    if not submitted:
        return

    if "avg_amount_to_balance" in features and "avg_amount" in values and "avg_balance" in values:
        values["avg_amount_to_balance"] = values["avg_amount"] / max(values["avg_balance"], 1.0)
    if "online_share" in values and "atm_share" in values and values["online_share"] + values["atm_share"] > 1.0001:
        st.error("Proporsi Online + ATM tidak boleh lebih dari 100% (sisanya kanal Cabang).")
        return

    row = pd.DataFrame([values])[features]
    seg = int(seg_pipe.predict(row)[0])

    prep = seg_pipe.named_steps["prep"]
    centers = seg_pipe.named_steps["km"].cluster_centers_
    x_t = prep.transform(row)[0]
    dist = np.linalg.norm(centers - x_t, axis=1)
    order = np.argsort(dist)
    ratio = float(dist[order[0]] / dist[order[1]]) if len(order) > 1 and dist[order[1]] > 0 else 0.0
    second = int(order[1]) if len(order) > 1 else seg

    if ratio < 0.6:
        clarity = "Penempatan jelas"
    elif ratio < 0.85:
        clarity = "Penempatan cukup jelas"
    else:
        clarity = f"Berada di perbatasan dengan {names.get(second, second)}"

    # ciri yang paling membedakan nasabah ini dari rata-rata, dihitung dalam satuan asli
    # (sama dengan angka yang ditampilkan, supaya kalimat dan angkanya selalu cocok)
    means = cust[features].mean()
    stds = cust[features].std().replace(0, np.nan)
    z_row = ((row.iloc[0] - means) / stds).fillna(0.0)
    traits = top_traits(row.iloc[0], means, z_row, features)

    items = "".join(
        f'<div class="trait"><div class="t">{esc(t["phrase"])}</div><div class="v">{esc(t["value"])} <span>(rata-rata {esc(t["avg"])})</span></div>'
        f'<div class="meter"><i style="width:{min(abs(t["z"]) / 1.5, 1) * 100:.0f}%;background:{colors[seg]}"></i></div></div>'
        for t in traits) or '<div class="none">Tidak ada ciri yang menonjol dibanding rata-rata nasabah.</div>'

    put(f"""
    <div class="result" style="border-left-color:{colors[seg]}">
    <div class="l">Paling mirip dengan</div>
    <div class="n">{esc(names.get(seg, seg))}</div>
    <div class="d">{esc(clarity)}</div>
    <div class="hint" style="margin-bottom:.4rem">Yang paling membedakan nasabah ini dari rata-rata:</div>
    {items}
    </div>
    """)

    with st.expander("Detail teknis"):
        clf_seg = int(clf.predict(row)[0])
        agree = "sepakat dengan" if clf_seg == seg else f"memprediksi {names.get(clf_seg, clf_seg)}, berbeda dari"
        put(f'<div class="hint">Jarak ke pusat segmen terdekat: {idn(dist[order[0]], 2)}. '
            f'Rasio jarak terdekat/kedua: {idn(ratio, 2)} (mendekati 1 = perbatasan; ambang 0,6 dan 0,85 adalah aturan praktis, bukan probabilitas). '
            f'Classifier pendamping {agree} KMeans.</div>')


def _supports_form_border():
    try:
        import inspect
        return "border" in inspect.signature(st.form).parameters
    except Exception:
        return False


def render_about(meta):
    level, label, explain = validity_level(meta)
    sil, null, ari = meta["silhouette"], meta["silhouette_null_mean"], meta["stability_ari"]
    rows = [
        ("Kekuatan pengelompokan", label, explain),
        ("Silhouette", idn(sil, 3), f"Seberapa terpisah antar segmen. Pada data yang diacak nilainya {idn(null, 3)}. Di atas 0,5 = kuat, di bawah 0,25 = lemah."),
        ("Stabilitas", idn(ari, 2), "Seberapa konsisten segmen jika data diambil ulang (skala 0 sampai 1; di atas 0,8 = stabil)."),
        ("Classifier pendamping", idn(meta["classifier_test_macro_f1"], 2),
         f"Model {meta['classifier']} untuk meniru segmen (macro-F1 pada data uji). Ini menunjukkan model bisa dipakai, bukan bukti segmen valid."),
    ]
    body = "".join(f'<tr><td class="k">{esc(k)}</td><td class="k">{esc(v)}</td><td class="e">{esc(e)}</td></tr>' for k, v, e in rows)
    put(f'<table class="about">{body}</table>')

    st.write("")
    st.markdown("**Batasan**")
    st.markdown(
        "- Data transaksi bersifat sintetis/terbatas, sehingga pola kelompok alami bisa lemah.\n"
        "- Tidak ada label fraud atau hasil bisnis, jadi dampak rekomendasi belum diuji (perlu A/B test).\n"
        "- Segmen ditetapkan dari seluruh periode data dan tidak diperbarui otomatis."
    )

    ver = meta.get("versions", {})
    if ver.get("sklearn") and ver["sklearn"] != sklearn.__version__:
        st.warning(f"Versi scikit-learn berbeda: model dilatih dengan {ver['sklearn']}, aplikasi memakai {sklearn.__version__}.")
    with st.expander("Data teknis lengkap (metadata.json)"):
        st.json(meta)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    missing = [f for f in REQUIRED_FILES if not (ARTIFACT_DIR / f).exists()]
    if missing:
        st.error("File berikut belum ada di folder `artifacts/`: " + ", ".join(missing) +
                 ". Jalankan Customer_Segmentation_Improved.ipynb sampai selesai terlebih dahulu.")
        st.stop()

    cust, profile, zprof, tx, meta, notes = load_tables()

    try:
        seg_pipe, clf = load_models()
    except Exception as exc:  # biasanya akibat beda versi scikit-learn antara pelatihan dan aplikasi
        trained = meta.get("versions", {}).get("sklearn", "(lihat metadata.json)")
        st.error(f"Model gagal dimuat. Penyebab paling umum: versi scikit-learn berbeda. "
                 f"Model dilatih dengan **{trained}**, sedangkan aplikasi memakai **{sklearn.__version__}**.")
        st.markdown("Samakan versinya di lingkungan Python yang menjalankan aplikasi, lalu jalankan ulang:")
        st.code(f"pip install scikit-learn=={trained}")
        st.caption(f"Detail teknis: {type(exc).__name__}: {exc}")
        st.stop()

    names = profile["segment_name"].to_dict()
    colors = seg_colors(names.keys())

    render_header(cust, tx, meta)
    tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Nasabah", "Simulasi", "Tentang"])
    with tab1:
        render_overview(cust, profile, zprof, tx, meta, notes, names, colors)
    with tab2:
        render_explorer(cust, meta, names, colors)
    with tab3:
        render_simulator(cust, profile, zprof, meta, seg_pipe, clf, names, colors)
    with tab4:
        render_about(meta)


main()
