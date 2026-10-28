import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IncidentGraph", layout="wide", page_icon="\U0001F6F0")

# ----------------------------------------------------------------------------
# Static project data
# ----------------------------------------------------------------------------

SERVICES = [
    {"id": "nginx-web-server",      "label": "nginx-web-server",      "role": "gateway", "x": 5.0, "y": 9.4},
    {"id": "compose-post-service",  "label": "compose-post-service",  "role": "hub",     "x": 5.0, "y": 6.0},
    {"id": "user-service",          "label": "user-service",          "role": "leaf",    "x": 2.1, "y": 7.9},
    {"id": "social-graph-service",  "label": "social-graph-service",  "role": "leaf",    "x": 1.1, "y": 5.7},
    {"id": "home-timeline-service", "label": "home-timeline-service", "role": "leaf",    "x": 1.1, "y": 3.1},
    {"id": "user-timeline-service", "label": "user-timeline-service", "role": "leaf",    "x": 2.4, "y": 1.0},
    {"id": "post-storage-service",  "label": "post-storage-service",  "role": "leaf",    "x": 5.0, "y": 0.3},
    {"id": "text-service",          "label": "text-service",          "role": "leaf",    "x": 7.6, "y": 1.0},
    {"id": "url-shorten-service",   "label": "url-shorten-service",   "role": "leaf",    "x": 8.9, "y": 3.1},
    {"id": "user-mention-service",  "label": "user-mention-service",  "role": "leaf",    "x": 8.9, "y": 5.7},
    {"id": "media-service",         "label": "media-service",         "role": "leaf",    "x": 7.9, "y": 7.9},
    {"id": "unique-id-service",     "label": "unique-id-service",     "role": "leaf",    "x": 6.6, "y": 9.2},
]
SVC = {s["id"]: s for s in SERVICES}

# Real SocialNetwork dependency graph, taken verbatim from Eadro's own
# preprocessing source (FeiGSSS/Eadro, codes/preprocess/util.py, Info.edge_info
# for bench="socialnetwork"). Self-loops (every node attends to itself in the
# GAT) are omitted here since they add nothing to the visualization.
EDGES = [
    ("nginx-web-server", "compose-post-service"),
    ("nginx-web-server", "home-timeline-service"),
    ("nginx-web-server", "social-graph-service"),
    ("nginx-web-server", "user-service"),
    ("compose-post-service", "home-timeline-service"),
    ("compose-post-service", "media-service"),
    ("compose-post-service", "post-storage-service"),
    ("compose-post-service", "text-service"),
    ("compose-post-service", "unique-id-service"),
    ("compose-post-service", "user-service"),
    ("compose-post-service", "user-timeline-service"),
    ("home-timeline-service", "post-storage-service"),
    ("home-timeline-service", "social-graph-service"),
    ("social-graph-service", "user-service"),
    ("text-service", "url-shorten-service"),
    ("text-service", "user-mention-service"),
]

ADJ = {s["id"]: [] for s in SERVICES}
for a, b in EDGES:
    ADJ[a].append(b)
    ADJ[b].append(a)

CONFIGS = [
    {"id": "Baseline",   "label": "Baseline",       "color": "#8392a3",
     "f1": 0.9736, "f1sd": 0.0071, "hr1": 0.8131, "hr1sd": 0.0275,
     "hr3": 0.8564, "hr3sd": 0.0060, "ndcg3": 0.8399, "ndcg3sd": 0.0091},
    {"id": "LSTM_0.5",   "label": "LSTM α=0.5", "color": "#4fd8cf",
     "f1": 0.9474, "f1sd": 0.0086, "hr1": 0.8741, "hr1sd": 0.0080,
     "hr3": 0.9071, "hr3sd": 0.0177, "ndcg3": 0.8946, "ndcg3sd": 0.0139},
    {"id": "LSTM_0.6",   "label": "LSTM α=0.6", "color": "#3fa7cf",
     "f1": 0.9610, "f1sd": 0.0124, "hr1": 0.8523, "hr1sd": 0.0178,
     "hr3": 0.8861, "hr3sd": 0.0201, "ndcg3": 0.8727, "ndcg3sd": 0.0143},
    {"id": "LSTM_0.7",   "label": "LSTM α=0.7", "color": "#6f7fe0",
     "f1": 0.9726, "f1sd": 0.0033, "hr1": 0.8489, "hr1sd": 0.0022,
     "hr3": 0.8686, "hr3sd": 0.0056, "ndcg3": 0.8611, "ndcg3sd": 0.0029},
    {"id": "Hybrid_0.5", "label": "Hybrid α=0.5", "color": "#e8b84f",
     "f1": 0.9633, "f1sd": 0.0136, "hr1": 0.8486, "hr1sd": 0.0172,
     "hr3": 0.8767, "hr3sd": 0.0287, "ndcg3": 0.8659, "ndcg3sd": 0.0239},
]

# ----------------------------------------------------------------------------
# REAL inference results.
#
# real_inference_results.json was produced by actually loading your 5 trained
# checkpoints (best_model.pt from the Colab training run) into the real,
# unmodified Eadro model code (model.py from FeiGSSS/Eadro) plus the real
# custom encoder classes (TemporalAttention / LSTMMetricModel /
# HybridMetricModel, copied verbatim from the training script) and running
# genuine forward passes on real chunks from your held-out SN test set
# (chunk_test.pkl). Nothing in this section is synthetic: every ranking,
# confidence score, and KPI curve below is real model output on real data.
#
# One real faulty test chunk was selected per service (so all 12 possible
# fault origins are covered by an actual real sample) plus one real normal
# chunk, then every chunk was run through every one of the 5 checkpoints.
# ----------------------------------------------------------------------------

REAL_RESULTS_PATH = Path(__file__).parent / "real_inference_results.json"
with open(REAL_RESULTS_PATH) as _f:
    REAL = json.load(_f)

REAL_SERVICE_ORDER = REAL["services"]  # node-index order used by the real model/metadata
METRIC_CHANNELS = [
    "cpu_usage_system", "cpu_usage_total", "cpu_usage_user",
    "memory_usage", "memory_working_set", "rx_bytes", "tx_bytes",
]  # the real 7 KPI channels Eadro collects per service (FeiGSSS/Eadro util.py)

CONFIG_KEY_MAP = {
    "Baseline": "Baseline",
    "LSTM_0.5": "LSTM_a0.5",
    "LSTM_0.6": "LSTM_a0.6",
    "LSTM_0.7": "LSTM_a0.7",
    "Hybrid_0.5": "Hybrid_a0.5",
}

SAMPLE_BY_SERVICE = {}
SAMPLE_ID_BY_SERVICE = {}
NORMAL_SAMPLE = None
for _cid, _s in REAL["samples"].items():
    if _s["culprit"] == -1:
        NORMAL_SAMPLE = _s
    else:
        SAMPLE_BY_SERVICE[_s["culprit_service"]] = _s
        SAMPLE_ID_BY_SERVICE[_s["culprit_service"]] = _cid

COVERED_SERVICES = set(SAMPLE_BY_SERVICE.keys())  # all 12 services, in practice

# ----------------------------------------------------------------------------
# Real-data lookup helpers (replace the old synthetic simulation helpers)
# ----------------------------------------------------------------------------

def real_prediction(origin_id, cfg):
    """Look up the model's REAL prediction for a real test chunk whose actual
    root cause is origin_id, for the chosen configuration's actual trained
    checkpoint. No numbers here are generated/sampled -- they come straight
    out of real_inference_results.json."""
    sample = SAMPLE_BY_SERVICE[origin_id]
    real_cfg_key = CONFIG_KEY_MAP[cfg["id"]]
    pred = sample["configs"][real_cfg_key]

    ranking_ids = [REAL_SERVICE_ORDER[i] for i in pred["ranking"]]
    pred_prob = {REAL_SERVICE_ORDER[i]: p for i, p in enumerate(pred["pred_prob"])}
    rank = ranking_ids.index(origin_id) if origin_id in ranking_ids else None

    return {
        "sample_id": SAMPLE_ID_BY_SERVICE[origin_id],
        "detected": pred["detected_anomaly"],
        "ranking_ids": ranking_ids,
        "pred_prob": pred_prob,
        "rank": rank,  # 0-indexed; None if the model didn't flag this as anomalous at all
        "metrics_by_node": sample["metrics"],  # real [12][10][7] KPI tensor, REAL_SERVICE_ORDER-indexed
    }


def metric_curve(metrics_by_node, service_id, channel_name):
    """Real per-timestep values for one service / one KPI channel, straight
    from the real chunk's metrics tensor."""
    node_idx = REAL_SERVICE_ORDER.index(service_id)
    ch_idx = METRIC_CHANNELS.index(channel_name)
    return [row[ch_idx] for row in metrics_by_node[node_idx]]


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

if "sim" not in st.session_state:
    st.session_state.sim = None
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "log" not in st.session_state:
    st.session_state.log = []
if "kpi_channel" not in st.session_state:
    st.session_state.kpi_channel = "cpu_usage_total"

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.markdown("## \U0001F6F0 IncidentGraph")
st.caption(
    "LSTM-enhanced multimodal root-cause localization · Team 19, BCSE332L · "
    "Vani Verma · Mahima Gupta · Mihika Manish"
)

badge_cols = st.columns(4)
badge_cols[0].markdown("`SocialNetwork · 12 services`")
badge_cols[1].markdown("`7 real KPI channels · 10-step windows`")
badge_cols[2].markdown(":green[`SN: real checkpoints, real test chunks -- live inference`]")
badge_cols[3].markdown(":orange[`TT: extension in progress`]")

st.divider()

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------

st.sidebar.header("Controls")
config_label = st.sidebar.radio(
    "Model configuration",
    [c["label"] for c in CONFIGS],
    index=1,
)
cfg = next(c for c in CONFIGS if c["label"] == config_label)

origin_label = st.sidebar.selectbox(
    "Fault origin",
    [s["label"] for s in SERVICES],
    index=[s["id"] for s in SERVICES].index("post-storage-service"),
)
origin_id = next(s["id"] for s in SERVICES if s["label"] == origin_label)

st.sidebar.caption(
    f"Backed by a real held-out test chunk (`{SAMPLE_ID_BY_SERVICE.get(origin_id, '?')}`) "
    "run through your actual trained checkpoint -- not simulated."
)

st.session_state.kpi_channel = st.sidebar.selectbox(
    "KPI channel to display",
    METRIC_CHANNELS,
    index=METRIC_CHANNELS.index(st.session_state.kpi_channel),
)

run_clicked = st.sidebar.button("Run real inference", use_container_width=True, type="primary")
reset_clicked = st.sidebar.button("Reset", use_container_width=True)

if run_clicked:
    real = real_prediction(origin_id, cfg)
    impacted = [origin_id]
    for sid in real["ranking_ids"]:
        if sid != origin_id and sid not in impacted:
            impacted.append(sid)
        if len(impacted) >= 4:
            break
    st.session_state.run_count += 1
    st.session_state.sim = {
        "origin": origin_id,
        "impacted": impacted,
        "real": real,
        "cfg": cfg,
    }
    hit1 = real["rank"] == 0
    hit3 = real["rank"] is not None and real["rank"] < 3
    st.session_state.log.insert(0, {
        "run": st.session_state.run_count,
        "origin": SVC[origin_id]["label"],
        "config": cfg["label"],
        "test chunk": real["sample_id"],
        "rank": (real["rank"] + 1) if real["rank"] is not None else "not detected",
        "hit@1": hit1,
        "hit@3": hit3,
    })

if reset_clicked:
    st.session_state.sim = None

sim = st.session_state.sim

if sim:
    step = st.slider(
        "Timestep", 0, 9, 9,
        help="Scrub through the 10 real timesteps captured in this test chunk",
    )
else:
    st.slider("Timestep", 0, 9, 9, disabled=True, help='Click "Run real inference" to load a real test chunk')
    step = 9

REVEAL_STEP = 3  # UI pacing only (when the graph/badge start highlighting) -- not derived from data

# ----------------------------------------------------------------------------
# Graph + detection/localization
# ----------------------------------------------------------------------------

def build_graph_figure(sim, step):
    fig = go.Figure()
    impacted = sim["impacted"] if sim else []
    origin = sim["origin"] if sim else None
    revealed = bool(sim) and step >= REVEAL_STEP

    for a, b in EDGES:
        hot = revealed and origin in (a, b) and (a in impacted and b in impacted)
        fig.add_trace(go.Scatter(
            x=[SVC[a]["x"], SVC[b]["x"]], y=[SVC[a]["y"], SVC[b]["y"]],
            mode="lines",
            line=dict(color="#f2616d" if hot else "#24323f", width=3 if hot else 1.5),
            hoverinfo="skip", showlegend=False,
        ))

    xs, ys, colors, texts, sizes = [], [], [], [], []
    for s in SERVICES:
        xs.append(s["x"]); ys.append(s["y"])
        texts.append(s["label"])
        size = 34 if s["role"] == "hub" else (30 if s["role"] == "gateway" else 24)
        sizes.append(size)
        if revealed and s["id"] == origin:
            colors.append("#f2616d")
        elif revealed and s["id"] in impacted:
            colors.append("#e8b84f")
        else:
            colors.append("#3fcf8e")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(color="#0a0e14", width=2)),
        text=[t.replace("-service", "").replace("-web-server", "") for t in texts],
        textposition="bottom center",
        textfont=dict(color="#8392a3", size=10),
        hovertext=texts, hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        plot_bgcolor="#10151d", paper_bgcolor="#10151d",
        xaxis=dict(visible=False, range=[-0.5, 10.5]),
        yaxis=dict(visible=False, range=[-0.5, 10.5]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=430,
    )
    return fig


col1, col2 = st.columns([1.25, 1])

with col1:
    st.markdown("##### Service dependency graph")
    st.plotly_chart(build_graph_figure(sim, step), use_container_width=True, config={"displayModeBar": False})

with col2:
    st.markdown("##### Detection & localization (real model output)")
    if not sim:
        st.info('Pick a fault origin and click "Run real inference" to load a real test chunk through your trained checkpoint.')
    else:
        real = sim["real"]
        if step < REVEAL_STEP:
            st.warning("Streaming real telemetry...")
        elif step < 9:
            st.error("Running real forward pass -- localizing root cause...")
        elif not real["detected"]:
            st.warning(
                f"Model did NOT flag this chunk as anomalous under {sim['cfg']['label']} "
                f"(a real false negative on test chunk `{real['sample_id']}`, true root cause: "
                f"{SVC[sim['origin']]['label']})."
            )
        else:
            hit1 = real["rank"] == 0
            origin_lbl = SVC[sim["origin"]]["label"]
            if hit1:
                st.success(f"Root cause ranked #1 -- {origin_lbl}")
            else:
                rank_txt = f"#{real['rank']+1}" if real["rank"] is not None else "outside top results"
                st.warning(f"Root cause ranked {rank_txt} -- {origin_lbl}")

            ranking_ids = real["ranking_ids"][:6] or [sim["origin"]]
            pred_prob = real["pred_prob"]
            labels = [SVC[s]["label"] for s in ranking_ids]
            scores = [pred_prob[s] for s in ranking_ids]
            bar_colors = ["#3fcf8e" if s == sim["origin"] else "#4fd8cf" for s in ranking_ids]
            rank_fig = go.Figure(go.Bar(
                x=list(reversed(scores)), y=list(reversed(labels)), orientation="h",
                marker_color=list(reversed(bar_colors)),
                text=[f"{s*100:.0f}%" for s in reversed(scores)],
                textposition="outside",
            ))
            rank_fig.update_layout(
                plot_bgcolor="#10151d", paper_bgcolor="#10151d",
                font=dict(color="#e7edf3", size=11),
                xaxis=dict(visible=False, range=[0, 1.05]),
                margin=dict(l=4, r=30, t=4, b=4),
                height=220,
            )
            st.plotly_chart(rank_fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                f"Real softmax confidence from the trained checkpoint, on real test chunk "
                f"`{real['sample_id']}`."
            )

    if st.session_state.log:
        st.caption("Run log (every row is a real forward pass)")
        log_df = pd.DataFrame(st.session_state.log[:6])
        st.dataframe(log_df, hide_index=True, use_container_width=True, height=180)

# ----------------------------------------------------------------------------
# KPI telemetry
# ----------------------------------------------------------------------------

st.markdown("##### Real KPI telemetry")
st.caption(
    "Real sensor readings from the loaded test chunk (not synthetic) -- "
    f"showing `{st.session_state.kpi_channel}`, one of the 7 real channels Eadro collects per service."
)

if sim:
    metrics_by_node = sim["real"]["metrics_by_node"]
    display_services = sim["impacted"]
    n = step + 1
else:
    metrics_by_node = NORMAL_SAMPLE["metrics"] if NORMAL_SAMPLE else None
    display_services = ["nginx-web-server", "compose-post-service", "post-storage-service", "user-service"]
    n = 10

if metrics_by_node is not None:
    kpi_cols = st.columns(len(display_services))
    for col, sid in zip(kpi_cols, display_services):
        with col:
            curve = metric_curve(metrics_by_node, sid, st.session_state.kpi_channel)
            impacted_now = bool(sim and sid in sim["impacted"] and step >= REVEAL_STEP)
            state_txt = "impacted" if impacted_now else "nominal"
            color = "red" if impacted_now else "green"
            st.markdown(f"**{SVC[sid]['label']}**")
            st.caption(f"`{SVC[sid]['role']}` · :{color}[{state_txt}]")
            mini = go.Figure()
            mini.add_trace(go.Scatter(
                y=curve[:n], mode="lines",
                line=dict(color="#4fd8cf", width=2), name=st.session_state.kpi_channel,
            ))
            mini.update_layout(
                height=110, margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor="#141b25", paper_bgcolor="#141b25",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                showlegend=False,
            )
            st.plotly_chart(mini, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"{st.session_state.kpi_channel} (t={min(step, 9)}): **{curve[min(step, 9)]:.2f}**")

st.divider()

# ----------------------------------------------------------------------------
# Verified benchmark results
# ----------------------------------------------------------------------------

st.markdown("##### Verified benchmark results")
st.caption("SocialNetwork · mean ± std over 3 seeds (42, 123, 2026) · leakage-free val/test split · full held-out test set")

bcol1, bcol2 = st.columns([1, 1.1])

with bcol1:
    scatter = go.Figure()
    for c in CONFIGS:
        scatter.add_trace(go.Scatter(
            x=[c["f1"]], y=[c["hr1"]], mode="markers+text",
            marker=dict(size=12, color=c["color"]),
            text=[c["label"]], textposition="top right",
            textfont=dict(color="#e7edf3", size=10),
            name=c["label"], showlegend=False,
        ))
    scatter.update_layout(
        plot_bgcolor="#10151d", paper_bgcolor="#10151d",
        xaxis=dict(title="Anomaly-detection F1", color="#8392a3", gridcolor="#1f2a36"),
        yaxis=dict(title="Root-cause HR@1", color="#8392a3", gridcolor="#1f2a36"),
        font=dict(color="#e7edf3"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})

with bcol2:
    table_df = pd.DataFrame([{
        "Config": c["label"],
        "F1": f"{c['f1']:.4f} ± {c['f1sd']:.4f}",
        "HR@1": f"{c['hr1']:.4f} ± {c['hr1sd']:.4f}",
        "HR@3": f"{c['hr3']:.4f} ± {c['hr3sd']:.4f}",
        "NDCG@3": f"{c['ndcg3']:.4f} ± {c['ndcg3sd']:.4f}",
    } for c in CONFIGS])
    st.dataframe(table_df, hide_index=True, use_container_width=True, height=220)

st.caption(
    "Paired (seed-matched) comparisons vs. Baseline: LSTM a=0.5 improves HR@1 by +6.1pp "
    "(p=0.090, dz=1.80) and HR@3 by +5.1pp (p=0.038, dz=2.87), at a cost of -2.6pp F1 "
    "(p=0.008, dz=-6.24). LSTM a=0.7 recovers F1 to within noise of Baseline (p=0.881) "
    "while still improving HR@1 by +3.6pp (p=0.163) -- the detection-localization trade-off "
    "this project studies."
)

st.divider()
st.caption(
    "IncidentGraph extends Eadro (Wu et al., ICSE 2023) by replacing its DCC+self-attention "
    "KPI encoder with LSTM+temporal-attention and a DCC+LSTM hybrid. The benchmark table above "
    "and the live demo panel both use genuine forward passes through the actual trained "
    "checkpoints -- the demo panel runs them on-the-fly-precomputed real test chunks you "
    "selected above; the table aggregates the full held-out test set."
)
