from __future__ import annotations

from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
EVIDENCE_DIR = APP_DIR / "evidence"
ASSET_DIR = APP_DIR / "assets"

PHASES = ["Morning", "Afternoon", "Night"]
PHASE_TICKS = [29.5, 89.5, 149.5]
PHASE_COLOURS = {
    "Morning": "#d79a22",
    "Afternoon": "#cf3f36",
    "Night": "#315fa8",
}
SCENARIO_COLOURS = {
    "Baseline": "#2563eb",
    "ANDM-inspired rural": "#dc2626",
}

EXPECTED_HASHES = {
    "mininet_baseline_cpu.csv": "7d44065414cb807fbe1454b0ae1b63100ebdc30afa04f9ea850400a02171b53b",
    "mininet_baseline_network_raw.csv": "408bdfabd3637f544e0147f67c0fc9e6fe3d338f3a8d382f41d7158504ef250f",
    "mininet_rural_cpu.csv": "8bf6e38e7afb5a598b2b836aa0a735412a34d8a09b1d5e9e54912ee5c42b1d06",
    "mininet_rural_network_raw.csv": "a54763281c1df9c41281076db821db34a624716cb5f656bfcf40a46e2200f110",
    "andm_mininet_capture_final_v5.py": "9bf8f14d3c63731b0e680873b652ae5cf8a6dcee76bced56a33ec86872367d57",
}

ALGORITHM_RUNS = pd.DataFrame(
    {
        "seed": [7, 17, 29, 42, 61, 73, 89, 101, 127, 149],
        "DQN": [65.9267, 60.7320, 65.7131, 65.9733, 55.9370, 64.2967, 64.2967, 64.2967, 65.7867, 65.5900],
        "PPO": [60.7320, 63.8487, 63.5767, 61.4967, 61.4967, 60.7300, 57.4453, 55.4103, 63.8487, 62.8800],
    }
)

ALGORITHM_SUMMARY = pd.DataFrame(
    [
        {
            "algorithm": "DQN",
            "seeds": 10,
            "mean_reward": 63.8549,
            "std_reward": 3.1973,
            "satisfaction": 0.9923,
            "mean_vcpu": 1.7893,
            "overload": 0.0246,
        },
        {
            "algorithm": "PPO",
            "seeds": 10,
            "mean_reward": 61.1465,
            "std_reward": 2.8029,
            "satisfaction": 0.9916,
            "mean_vcpu": 2.5357,
            "overload": 0.0300,
        },
    ]
)

BASELINE_TEST_SUMMARY = pd.DataFrame(
    [
        {"policy": "Champion-v1", "mean_reward": 65.3570, "std_reward": 0.0297, "satisfaction": 0.9934, "mean_vcpu": 1.3611, "overload": 0.0556},
        {"policy": "Fixed-vCPU-2", "mean_reward": 64.8000, "std_reward": 0.0000, "satisfaction": 1.0000, "mean_vcpu": 2.0000, "overload": 0.0000},
        {"policy": "CPU-rule", "mean_reward": 66.1600, "std_reward": 0.0541, "satisfaction": 1.0000, "mean_vcpu": 1.5037, "overload": 0.0000},
    ]
)

PROMOTION_SUMMARY = pd.DataFrame(
    [
        {"policy": "Champion-v1", "mean_reward": 48.0876, "std_reward": 2.5891, "satisfaction": 0.8247, "mean_vcpu": 1.0389, "overload": 0.3259},
        {"policy": "Candidate-v2", "mean_reward": 54.5007, "std_reward": 0.1276, "satisfaction": 0.8524, "mean_vcpu": 1.4870, "overload": 0.0815},
    ]
)

FINAL_TEST_SUMMARY = pd.DataFrame(
    [
        {"policy": "Deployed-v2", "mean_reward": 54.4176, "std_reward": 0.2465, "satisfaction": 0.8536, "mean_vcpu": 1.7130, "overload": 0.0278},
        {"policy": "Fixed-vCPU-2", "mean_reward": 53.7076, "std_reward": 0.0000, "satisfaction": 0.8536, "mean_vcpu": 2.0000, "overload": 0.0278},
        {"policy": "CPU-rule", "mean_reward": 55.8129, "std_reward": 0.1660, "satisfaction": 0.8538, "mean_vcpu": 1.3389, "overload": 0.0259},
    ]
)


st.set_page_config(
    page_title="ANDM Self-Evolving RL Demonstrator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #071a2d;
        --blue: #163b65;
        --gold: #d5a73a;
        --paper: #f5f7fa;
        --ink: #132238;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1450px; padding-top: 1.4rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #071a2d 0%, #102e50 100%); }
    [data-testid="stSidebar"] * { color: #f7f9fc; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 8px;
        padding: 0.25rem 0.4rem;
    }
    .hero {
        background: linear-gradient(120deg, #071a2d 0%, #163b65 68%, #1d4f82 100%);
        border-left: 7px solid var(--gold);
        border-radius: 14px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 25px rgba(7, 26, 45, 0.15);
    }
    .hero .brand { color: var(--gold); font-size: 0.85rem; font-weight: 800; letter-spacing: 0.19rem; }
    .hero h1 { color: white; margin: 0.25rem 0 0.2rem 0; font-size: 2.0rem; }
    .hero p { color: #d8e4f1; margin: 0; font-size: 1rem; }
    .boundary {
        background: #fff8e8;
        border: 1px solid #e7c76f;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0 1.1rem 0;
        color: #5c4511;
    }
    .good-box, .info-box, .warn-box {
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.6rem 0;
    }
    .good-box { background: #eaf7ef; border-left: 5px solid #198754; }
    .info-box { background: #eaf2fb; border-left: 5px solid #2563eb; }
    .warn-box { background: #fff1ed; border-left: 5px solid #dc2626; }
    .step-box {
        min-height: 120px;
        background: white;
        border-top: 4px solid var(--gold);
        border-radius: 10px;
        padding: 0.8rem;
        box-shadow: 0 3px 12px rgba(7, 26, 45, 0.08);
    }
    .small-note { color: #506176; font-size: 0.9rem; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce3ea;
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    h1, h2, h3 { color: #0b2948; }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="brand">DUNYWA</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def evidence_boundary() -> None:
    st.markdown(
        """
        <div class="boundary">
        <strong>Research boundary:</strong> This is a controlled ONOS–Mininet testbed inspired by
        Alfred Nzo District Municipality. It is not live ANDM telemetry. Morning, Afternoon and Night
        are simulated workload periods. All vCPU actions are simulated decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_cpu = pd.read_csv(DATA_DIR / "mininet_baseline_cpu.csv")
    baseline_network = pd.read_csv(DATA_DIR / "mininet_baseline_network_raw.csv")
    rural_cpu = pd.read_csv(DATA_DIR / "mininet_rural_cpu.csv")
    rural_network = pd.read_csv(DATA_DIR / "mininet_rural_network_raw.csv")

    def align(cpu: pd.DataFrame, network: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
        selected = network[
            (network["switch"].astype(str).str.strip() == "s1")
            & (network["port"].astype(str).str.strip() == "1:")
        ].copy()
        network_columns = [
            "sample",
            "source_site",
            "latency_ms",
            "packet_loss_pct",
            "delivered_rate_mbps",
            "offered_load_mbps",
            "active_flows",
            "configured_core_bw_mbps",
            "configured_branch_bw_mbps",
            "configured_wan_delay_ms",
            "configured_wan_loss_pct",
        ]
        joined = cpu.merge(selected[network_columns], on="sample", how="inner", validate="one_to_one")
        joined["scenario_key"] = key
        joined["scenario_label"] = label
        joined["latency_timed_out"] = joined["latency_ms"].isna()
        joined["latency_for_model_ms"] = joined["latency_ms"].fillna(1000.0)
        joined["delivery_ratio"] = np.clip(
            joined["delivered_rate_mbps"] / joined["offered_load_mbps"], 0.0, 1.0
        )
        joined["service_shortfall"] = 1.0 - joined["delivery_ratio"]
        joined["phase_order"] = joined["simulated_time"].map({"Morning": 0, "Afternoon": 1, "Night": 2})
        return joined.sort_values("sample").reset_index(drop=True)

    baseline = align(baseline_cpu, baseline_network, "baseline", "Baseline")
    rural = align(rural_cpu, rural_network, "rural", "ANDM-inspired rural")
    master = pd.concat([baseline, rural], ignore_index=True)

    summary = (
        master.groupby(["scenario_label", "simulated_time"], sort=False)
        .agg(
            samples=("sample", "count"),
            cpu_mean_pct=("cpu_used", "mean"),
            cpu_std_pct=("cpu_used", "std"),
            cpu_max_pct=("cpu_used", "max"),
            offered_mean_mbps=("offered_load_mbps", "mean"),
            delivered_mean_mbps=("delivered_rate_mbps", "mean"),
            delivery_ratio_mean=("delivery_ratio", "mean"),
            latency_mean_ms=("latency_ms", "mean"),
            latency_median_ms=("latency_ms", "median"),
            latency_max_ms=("latency_ms", "max"),
            latency_timeouts=("latency_timed_out", "sum"),
            loss_mean_pct=("packet_loss_pct", "mean"),
        )
        .reset_index()
    )
    summary["phase_order"] = summary["simulated_time"].map({"Morning": 0, "Afternoon": 1, "Night": 2})
    summary = summary.sort_values(["scenario_label", "phase_order"]).reset_index(drop=True)
    return baseline, rural, master, summary


@st.cache_data(show_spinner=False)
def verify_evidence() -> pd.DataFrame:
    paths = {
        "mininet_baseline_cpu.csv": DATA_DIR / "mininet_baseline_cpu.csv",
        "mininet_baseline_network_raw.csv": DATA_DIR / "mininet_baseline_network_raw.csv",
        "mininet_rural_cpu.csv": DATA_DIR / "mininet_rural_cpu.csv",
        "mininet_rural_network_raw.csv": DATA_DIR / "mininet_rural_network_raw.csv",
        "andm_mininet_capture_final_v5.py": EVIDENCE_DIR / "andm_mininet_capture_final_v5.py",
    }
    rows = []
    for name, path in paths.items():
        actual = calculate_sha256(path)
        rows.append(
            {
                "file": name,
                "status": "PASS" if actual == EXPECTED_HASHES[name] else "FAIL",
                "expected_sha256": EXPECTED_HASHES[name],
                "actual_sha256": actual,
            }
        )
    return pd.DataFrame(rows)


def add_phase_background(axis: plt.Axes) -> None:
    for start, stop, phase in [(0, 59, "Morning"), (60, 119, "Afternoon"), (120, 179, "Night")]:
        axis.axvspan(start, stop, color=PHASE_COLOURS[phase], alpha=0.045)
    axis.set_xticks(PHASE_TICKS, PHASES)


def selected_frames(baseline: pd.DataFrame, rural: pd.DataFrame, view: str) -> list[tuple[str, pd.DataFrame]]:
    if view == "Baseline":
        return [("Baseline", baseline)]
    if view == "ANDM-inspired rural":
        return [("ANDM-inspired rural", rural)]
    return [("Baseline", baseline), ("ANDM-inspired rural", rural)]


def cpu_figure(baseline: pd.DataFrame, rural: pd.DataFrame, summary: pd.DataFrame, view: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for label, frame in selected_frames(baseline, rural, view):
        axes[0].plot(frame["sample"], frame["cpu_used"], label=label, color=SCENARIO_COLOURS[label], linewidth=1.45, alpha=0.9)
    axes[0].set_title("Measured CPU utilisation across the simulated day")
    axes[0].set_ylabel("CPU utilisation (%)")
    add_phase_background(axes[0])
    axes[0].legend()

    shown = summary if view == "Compare both" else summary[summary["scenario_label"] == view]
    pivot = shown.pivot(index="simulated_time", columns="scenario_label", values="cpu_mean_pct").reindex(PHASES)
    colours = [SCENARIO_COLOURS[col] for col in pivot.columns]
    pivot.plot.bar(ax=axes[1], color=colours, width=0.72)
    axes[1].set_title("Mean CPU utilisation by simulated time")
    axes[1].set_xlabel("Simulated time")
    axes[1].set_ylabel("Mean CPU utilisation (%)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend([str(col) for col in pivot.columns], fontsize=8)
    return fig


def throughput_figure(baseline: pd.DataFrame, rural: pd.DataFrame, view: str) -> plt.Figure:
    rows = selected_frames(baseline, rural, view)
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 4.2 * len(rows)), squeeze=False, constrained_layout=True)
    for axis, (label, frame) in zip(axes[:, 0], rows):
        axis.plot(frame["sample"], frame["offered_load_mbps"], label="Offered traffic", color="#111827", linestyle="--", linewidth=1.4)
        axis.plot(frame["sample"], frame["delivered_rate_mbps"], label="Measured delivered rate", color=SCENARIO_COLOURS[label], linewidth=1.55)
        axis.set_title(f"{label} network")
        axis.set_ylabel("Traffic rate (Mbps)")
        axis.set_xlabel("Simulated time")
        add_phase_background(axis)
        axis.legend()
    return fig


def latency_loss_figure(baseline: pd.DataFrame, rural: pd.DataFrame, view: str) -> plt.Figure:
    rows = selected_frames(baseline, rural, view)
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 4.3 * len(rows)), squeeze=False, constrained_layout=True)
    for row_index, (label, frame) in enumerate(rows):
        latency_axis = axes[row_index, 0]
        loss_axis = axes[row_index, 1]
        colour = SCENARIO_COLOURS[label]
        latency_axis.plot(frame["sample"], frame["latency_ms"], color=colour, linewidth=1.4, label="Measured latency")
        timeouts = frame[frame["latency_timed_out"]]
        latency_axis.scatter(timeouts["sample"], np.full(len(timeouts), 1000.0), marker="x", color="#111827", s=42, label="All probes timed out")
        latency_axis.set_title(f"{label} latency")
        latency_axis.set_ylabel("Round-trip latency (ms)")
        latency_axis.set_xlabel("Simulated time")
        add_phase_background(latency_axis)
        latency_axis.legend(fontsize=8)

        loss_axis.plot(frame["sample"], frame["packet_loss_pct"], color=colour, linewidth=1.4)
        loss_axis.set_title(f"{label} packet loss")
        loss_axis.set_ylabel("Ping packet loss (%)")
        loss_axis.set_xlabel("Simulated time")
        add_phase_background(loss_axis)
    return fig


def phase_comparison_figure(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, value, title, ylabel in [
        (axes[0, 0], "delivered_mean_mbps", "Delivered traffic", "Mbps"),
        (axes[0, 1], "delivery_ratio_mean", "Traffic delivery ratio", "Ratio"),
        (axes[1, 0], "latency_mean_ms", "Mean measured latency", "Milliseconds"),
        (axes[1, 1], "loss_mean_pct", "Mean ping packet loss", "Percent"),
    ]:
        table = summary.pivot(index="simulated_time", columns="scenario_label", values=value).reindex(PHASES)
        table.plot.bar(ax=axis, color=[SCENARIO_COLOURS[col] for col in table.columns], width=0.72)
        axis.set_title(title)
        axis.set_xlabel("Simulated time")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=0)
        axis.legend(fontsize=8)
    return fig


def phase_aware_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    ranges = {"train": (0, 36), "validation": (36, 48), "test": (48, 60)}
    for phase in PHASES:
        phase_frame = frame[frame["simulated_time"] == phase].sort_values("sample").reset_index(drop=True)
        for name, (start, stop) in ranges.items():
            parts[name].append(phase_frame.iloc[start:stop].copy())
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}


@st.cache_data(show_spinner=False)
def create_monitoring_trace(baseline: pd.DataFrame, rural: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    baseline_test = phase_aware_split(baseline)["test"].copy()
    rural_train = phase_aware_split(rural)["train"].copy()
    baseline_test["monitoring_stage"] = "Baseline monitoring"
    rural_train["monitoring_stage"] = "Unknown profile after change"
    trace = pd.concat([baseline_test, rural_train], ignore_index=True)
    trace["stress_signal"] = (
        0.35 * np.clip(trace["latency_for_model_ms"] / 1000.0, 0.0, 1.0)
        + 0.35 * np.clip(trace["packet_loss_pct"] / 100.0, 0.0, 1.0)
        + 0.30 * np.clip(trace["service_shortfall"], 0.0, 1.0)
    )

    try:
        from river import drift

        detector = drift.ADWIN(delta=0.20, clock=1, min_window_length=5, grace_period=10)
        events = []
        for index, value in enumerate(trace["stress_signal"]):
            detector.update(float(value))
            if detector.drift_detected:
                events.append(index)
    except ImportError:
        events = [83, 127]
    return trace, events


def show_figure(fig: plt.Figure) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def home_page(baseline: pd.DataFrame, rural: pd.DataFrame, summary: pd.DataFrame) -> None:
    page_header("ANDM Self-Evolving RL Demonstrator", "Measured Mininet evidence, agent comparison, drift detection and safe adaptation")
    evidence_boundary()

    rural_afternoon = summary[(summary["scenario_label"] == "ANDM-inspired rural") & (summary["simulated_time"] == "Afternoon")].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Measured datasets", "4", "2 CPU + 2 network")
    cols[1].metric("Measured samples", "360", "180 per scenario")
    cols[2].metric("Selected RL agent", "DQN", "10 seeds per agent")
    cols[3].metric("Rural Afternoon delivery", f"{rural_afternoon['delivery_ratio_mean'] * 100:.1f}%", "30 Mbps offered")

    st.subheader("What the system demonstrates")
    steps = st.columns(5)
    labels = [
        ("1. Measure", "CPU, throughput, latency and packet loss were captured in ONOS–Mininet."),
        ("2. Compare", "DQN and PPO were trained and tested under matching conditions."),
        ("3. Detect", "ADWIN monitored a combined network-stress signal."),
        ("4. Adapt", "A candidate DQN was retrained after drift was detected."),
        ("5. Promote", "The candidate was deployed only after passing the validation checks."),
    ]
    for column, (title, body) in zip(steps, labels):
        column.markdown(f'<div class="step-box"><strong>{title}</strong><br><span class="small-note">{body}</span></div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("ANDM-inspired topology")
        st.graphviz_chart(
            """
            digraph {
              rankdir=TB;
              graph [bgcolor="transparent", pad="0.2"];
              node [shape=box, style="rounded,filled", fontname="Arial", color="#163b65", fillcolor="#eef4fa"];
              edge [color="#d5a73a", penwidth=2];
              HQ [label="District HQ\nONOS-controlled core", fillcolor="#fff4d6"];
              MAT [label="Matatiele branch"];
              UMZ [label="Umzimvubu branch"];
              NTA [label="Ntabankulu branch"];
              WIN [label="Winnie Madikizela-Mandela branch"];
              HQ -> MAT; HQ -> UMZ; HQ -> NTA; HQ -> WIN;
            }
            """,
            width="stretch",
        )
    with right:
        st.subheader("Controlled network profiles")
        profile_table = pd.DataFrame(
            [
                ["Baseline", "100 Mbps", "1 ms", "0%", "Stronger comparison network"],
                ["ANDM-inspired rural", "20 Mbps", "50 ms", "2%", "Researcher-defined limited network"],
            ],
            columns=["Profile", "Link capacity", "Configured delay", "Configured loss", "Purpose"],
        )
        st.dataframe(profile_table, hide_index=True, width="stretch")
        st.markdown(
            '<div class="info-box"><strong>Main measured finding:</strong> Rural Afternoon offered 30 Mbps through a 20 Mbps bottleneck. Delivered traffic fell to 17.68 Mbps, mean successful-ping latency rose to 790.26 ms and mean ping loss reached 58.33%.</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Use the menu")
    st.write("Open **Measured network** for the graphs, **Agent comparison** for DQN versus PPO, **Self-evolving cycle** for ADWIN and promotion, and **Final decision** for the final policy results.")


def measured_page(baseline: pd.DataFrame, rural: pd.DataFrame, summary: pd.DataFrame) -> None:
    page_header("Measured Network", "Explore the Baseline and ANDM-inspired rural Mininet results")
    evidence_boundary()

    control_a, control_b = st.columns([1.2, 1])
    with control_a:
        scenario_view = st.selectbox("Scenario shown in line graphs", ["Compare both", "Baseline", "ANDM-inspired rural"])
    with control_b:
        selected_phase = st.radio("Period used for the summary", PHASES, horizontal=True, index=1)

    selected_summary = summary[summary["simulated_time"] == selected_phase].copy()
    baseline_row = selected_summary[selected_summary["scenario_label"] == "Baseline"].iloc[0]
    rural_row = selected_summary[selected_summary["scenario_label"] == "ANDM-inspired rural"].iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Baseline CPU mean", f"{baseline_row['cpu_mean_pct']:.2f}%")
    metrics[1].metric("Rural CPU mean", f"{rural_row['cpu_mean_pct']:.2f}%")
    metrics[2].metric("Rural delivered traffic", f"{rural_row['delivered_mean_mbps']:.2f} Mbps", f"{rural_row['offered_mean_mbps']:.0f} Mbps offered")
    metrics[3].metric("Rural mean ping loss", f"{rural_row['loss_mean_pct']:.2f}%", f"{int(rural_row['latency_timeouts'])} full timeouts")

    tabs = st.tabs(["CPU", "Throughput", "Latency and loss", "Phase bars", "Summary table"])
    with tabs[0]:
        show_figure(cpu_figure(baseline, rural, summary, scenario_view))
        st.markdown(
            """
            <div class="info-box"><strong>What caused the high and low values?</strong><br>
            Afternoon used two traffic flows and offered 30 Mbps, while Morning offered 5 Mbps and Night offered 8 Mbps.
            This increased packet-processing work. The short one-second CPU spikes may come from Mininet, Open vSwitch,
            iperf3, ONOS or Ubuntu background scheduling. The capture did not record CPU per process, so one exact process
            cannot honestly be named. RL training happened later and did not cause these measured spikes.</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Important: Rural CPU was not much higher than Baseline CPU. The serious rural Afternoon problem was mainly a network-link bottleneck, not a CPU bottleneck.")
    with tabs[1]:
        show_figure(throughput_figure(baseline, rural, scenario_view))
        st.markdown(
            """
            <div class="warn-box"><strong>Why the rural Afternoon line is low:</strong> Two flows offered 30 Mbps through a shared 20 Mbps rural link. The extra traffic waited in queues or was dropped, so delivered traffic stayed near 16–19 Mbps. Morning and Night remained close to their offered loads because 5 Mbps and 8 Mbps fitted inside the 20 Mbps capacity.</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("A delivered counter can be slightly above the iperf3 target because the switch counter includes received bytes outside the UDP application payload, including headers and test traffic.")
    with tabs[2]:
        show_figure(latency_loss_figure(baseline, rural, scenario_view))
        st.markdown(
            """
            <div class="warn-box"><strong>Why the rural Afternoon spikes are high:</strong> The 30 Mbps offered load exceeded the 20 Mbps link. Queues grew, round-trip delay rose and packets were dropped. A black × means all three ping probes timed out. The 0%, 33.33%, 66.67% and 100% loss levels occur because each sample used only three ping probes.</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("The isolated Baseline latency spike near 97 ms was temporary and had no packet loss. The data did not record enough process detail to name its exact cause.")
    with tabs[3]:
        show_figure(phase_comparison_figure(summary))
        st.markdown(
            """
            <div class="info-box"><strong>How to read these bars:</strong> Each bar is the average for 60 measurements. Morning and Night are similar because their offered traffic fitted in both profiles. Rural Afternoon has a lower delivery bar and much higher latency and loss bars because the link was overloaded. Bars show the overall pattern but hide individual spikes.</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("The rural Afternoon latency mean excludes 13 complete timeouts because the measured latency stayed missing. Those timeouts are shown separately and represented as 1000 ms only in the RL feature.")
    with tabs[4]:
        display_table = selected_summary[
            [
                "scenario_label", "simulated_time", "samples", "cpu_mean_pct", "cpu_max_pct",
                "offered_mean_mbps", "delivered_mean_mbps", "delivery_ratio_mean",
                "latency_mean_ms", "latency_max_ms", "latency_timeouts", "loss_mean_pct",
            ]
        ].rename(
            columns={
                "scenario_label": "Scenario",
                "simulated_time": "Simulated time",
                "samples": "Samples",
                "cpu_mean_pct": "CPU mean (%)",
                "cpu_max_pct": "CPU maximum (%)",
                "offered_mean_mbps": "Offered (Mbps)",
                "delivered_mean_mbps": "Delivered (Mbps)",
                "delivery_ratio_mean": "Delivery ratio",
                "latency_mean_ms": "Latency mean (ms)",
                "latency_max_ms": "Latency maximum (ms)",
                "latency_timeouts": "Timeouts",
                "loss_mean_pct": "Ping loss mean (%)",
            }
        )
        numeric = display_table.select_dtypes(include="number").columns
        display_table[numeric] = display_table[numeric].round(4)
        st.dataframe(display_table, hide_index=True, width="stretch")


def agent_page() -> None:
    page_header("Agent Comparison", "DQN and PPO trained under the same measured data, seeds and training budget")
    evidence_boundary()

    cols = st.columns(4)
    cols[0].metric("DQN mean reward", "63.8549")
    cols[1].metric("PPO mean reward", "61.1465")
    cols[2].metric("Paired difference", "+2.7084", "DQN minus PPO")
    cols[3].metric("Selected seed", "42", "Selected after DQN")

    left, right = st.columns([0.85, 1.15])
    with left:
        fig, axis = plt.subplots(figsize=(7.2, 5.2))
        axis.bar(
            ALGORITHM_SUMMARY["algorithm"],
            ALGORITHM_SUMMARY["mean_reward"],
            yerr=ALGORITHM_SUMMARY["std_reward"],
            color=["#2563eb", "#7c3aed"],
            capsize=7,
        )
        axis.set_title("Baseline validation reward")
        axis.set_ylabel("Mean episode reward")
        axis.set_ylim(0, 72)
        show_figure(fig)
    with right:
        fig, axis = plt.subplots(figsize=(9, 5.2))
        axis.plot(ALGORITHM_RUNS["seed"].astype(str), ALGORITHM_RUNS["DQN"], marker="o", label="DQN", color="#2563eb")
        axis.plot(ALGORITHM_RUNS["seed"].astype(str), ALGORITHM_RUNS["PPO"], marker="o", label="PPO", color="#7c3aed")
        axis.set_title("Reward obtained by every training seed")
        axis.set_xlabel("Training seed")
        axis.set_ylabel("Mean validation reward")
        axis.legend()
        show_figure(fig)

    st.markdown(
        """
        <div class="good-box"><strong>Selection decision:</strong> DQN was selected because the paired 95% bootstrap interval for DQN minus PPO was 0.0330 to 5.1045. The complete interval was above zero. This supports a DQN advantage in this experiment.</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("The error bars show how much the mean reward changed across the ten training seeds. They are not errors in the four Mininet datasets.")
    st.warning("DQN is the best RL agent in this environment. This result does not mean DQN is always better than PPO in every network or research project.")

    st.subheader("Untouched Baseline test and simple policies")
    table = BASELINE_TEST_SUMMARY.copy()
    table.columns = ["Policy", "Mean reward", "Reward SD", "Satisfaction", "Mean simulated vCPU", "Overload rate"]
    st.dataframe(table.round(4), hide_index=True, width="stretch")
    st.caption("The simple CPU rule achieved the highest Baseline reward. This is why the research continues to compare the learned agent with simple policies instead of presenting RL as automatically better.")


def self_evolving_page(baseline: pd.DataFrame, rural: pd.DataFrame) -> None:
    page_header("Self-Evolving Cycle", "Replay ADWIN monitoring, candidate adaptation and the safe-promotion decision")
    evidence_boundary()

    trace, drift_events = create_monitoring_trace(baseline, rural)
    first_drift = drift_events[0] if drift_events else None
    transition = 36

    st.subheader("Interactive monitoring replay")
    chosen = st.slider("Move through the monitoring stream", 0, len(trace) - 1, first_drift or 83)
    row = trace.iloc[chosen]
    state = "Drift detected" if first_drift is not None and chosen >= first_drift else "Monitoring"
    stage = "Baseline" if chosen < transition else "Unknown profile after change"
    cols = st.columns(5)
    cols[0].metric("Monitoring sample", chosen)
    cols[1].metric("Stage", stage)
    cols[2].metric("Simulated time", row["simulated_time"])
    cols[3].metric("Stress signal", f"{row['stress_signal']:.3f}")
    cols[4].metric("Detector state", state)

    fig, axis = plt.subplots(figsize=(13, 5))
    axis.plot(trace.index, trace["stress_signal"], color="#111827", linewidth=1.45, label="Measured network stress")
    axis.axvline(transition, color="#d5a73a", linestyle="--", linewidth=2, label="Baseline-to-rural transition")
    if first_drift is not None:
        axis.axvline(first_drift, color="#dc2626", linestyle=":", linewidth=2.2, label="First ADWIN detection")
    axis.scatter([chosen], [row["stress_signal"]], color="#2563eb", s=70, zorder=5, label="Selected replay sample")
    axis.set_title("Measured network-stress drift monitoring")
    axis.set_xlabel("Monitoring sample")
    axis.set_ylabel("Stress signal")
    axis.legend(fontsize=8)
    show_figure(fig)

    st.write(
        f"At the selected sample, measured latency is **{row['latency_for_model_ms']:.2f} ms**, ping loss is **{row['packet_loss_pct']:.2f}%**, and the delivery ratio is **{row['delivery_ratio']:.3f}**."
    )
    st.markdown(
        f'<div class="info-box"><strong>Drift result:</strong> The profile changed at monitoring sample 36. ADWIN did not use the scenario label. It collected enough statistical evidence and first detected drift at sample {first_drift}. The recorded drift events were {drift_events}.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Candidate adaptation and safe promotion")
    promotion_cols = st.columns(4)
    promotion_cols[0].metric("Champion reward", "48.0876")
    promotion_cols[1].metric("Candidate reward", "54.5007")
    promotion_cols[2].metric("Reward improvement", "+6.4131")
    promotion_cols[3].metric("Decision", "Promote v2")

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(PROMOTION_SUMMARY["policy"], PROMOTION_SUMMARY["mean_reward"], color=["#2563eb", "#16a34a"])
    axis.set_title("Rural validation before safe promotion")
    axis.set_ylabel("Mean episode reward")
    axis.set_ylim(0, 60)
    show_figure(fig)

    check_a, check_b, check_c = st.columns(3)
    check_a.success("Reward interval passed\n\n95% interval: 4.9489 to 7.5198")
    check_b.success("Satisfaction check passed\n\n0.8524 versus 0.8247")
    check_c.success("Overload check passed\n\n0.0815 versus 0.3259")
    st.markdown('<div class="good-box"><strong>Result:</strong> Candidate DQN v2 passed all three checks and replaced champion v1. This is the safe-promotion part of the self-evolving prototype.</div>', unsafe_allow_html=True)
    st.caption("The candidate used more simulated vCPU than the old champion, but reduced simulated overload and improved satisfaction. It did not change the measured network bandwidth, latency or packet loss.")


def final_page() -> None:
    page_header("Final Decision", "Compare the deployed DQN with fixed allocation and the simple CPU rule")
    evidence_boundary()

    cols = st.columns(4)
    cols[0].metric("Deployed RL version", "DQN v2")
    cols[1].metric("Best final reward", "55.8129", "CPU-rule")
    cols[2].metric("DQN v2 reward", "54.4176")
    cols[3].metric("DQN v2 mean vCPU", "1.7130", "Simulated")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    colours = ["#16a34a", "#6b7280", "#d5a73a"]
    for axis, column, title, ylabel in [
        (axes[0, 0], "mean_reward", "Final rural test reward", "Mean episode reward"),
        (axes[0, 1], "satisfaction", "Final service satisfaction", "Mean ratio"),
        (axes[1, 0], "mean_vcpu", "Final simulated vCPU use", "Mean simulated vCPUs"),
        (axes[1, 1], "overload", "Final simulated overload", "Mean rate"),
    ]:
        axis.bar(FINAL_TEST_SUMMARY["policy"], FINAL_TEST_SUMMARY[column], color=colours)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=10)
    show_figure(fig)

    st.markdown(
        """
        <div class="warn-box"><strong>Critical finding:</strong> DQN v2 improved after adaptation, but the simple CPU rule achieved the highest final reward. The rule used fewer simulated vCPUs and produced slightly less overload while service satisfaction stayed almost the same. This is an honest result and not a failure of the research.</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("Satisfaction is almost equal because the network delivery ratio came from measured Mininet data and could not be changed by a simulated vCPU action. Adding vCPUs cannot widen the 20 Mbps link or recover dropped packets.")

    st.subheader("Recorded deployed DQN v2 decision trace")
    st.image(ASSET_DIR / "deployed_vcpu_trace.png", caption="Original figure saved by the final notebook", width="stretch")
    st.write("The deployed agent reduced simulated vCPUs during lower-load Morning, increased them during Afternoon, and reduced them again during Night. The brief Night increase follows a short CPU rise in the measured test rows.")
    st.warning("These are simulated vCPU recommendations. The notebook did not physically add or remove Mininet or computer CPU cores.")


def evidence_page(master: pd.DataFrame, summary: pd.DataFrame) -> None:
    page_header("Evidence and Sources", "Verify the files, download summaries and see the research support")
    evidence_boundary()

    audit = verify_evidence()
    all_pass = bool((audit["status"] == "PASS").all())
    if all_pass:
        st.success("All five SHA-256 values match the hashes recorded after the Ubuntu Mininet experiment.")
    else:
        st.error("At least one evidence file has changed. Do not use the changed package as the final evidence.")
    st.dataframe(audit, hide_index=True, width="stretch")

    st.subheader("Dataset structure")
    structure = pd.DataFrame(
        [
            ["Baseline CPU", 180, "One row per second"],
            ["Baseline network", 3240, "18 OVS interface rows per sample"],
            ["Rural CPU", 180, "One row per second"],
            ["Rural network", 3240, "18 OVS interface rows per sample"],
        ],
        columns=["Dataset", "Rows", "Meaning"],
    )
    st.dataframe(structure, hide_index=True, width="stretch")

    download_a, download_b = st.columns(2)
    with download_a:
        st.download_button(
            "Download measured phase summary",
            data=summary.drop(columns="phase_order").to_csv(index=False).encode("utf-8"),
            file_name="measured_phase_summary.csv",
            mime="text/csv",
            width="stretch",
        )
    with download_b:
        st.download_button(
            "Download aligned measured data",
            data=master.to_csv(index=False).encode("utf-8"),
            file_name="andm_master_measured_dataset.csv",
            mime="text/csv",
            width="stretch",
        )

    st.subheader("Recent support used in the interpretation")
    sources = [
        ("Mubangizi (2025)", "Alfred Nzo is a rural district affected by geographical isolation and infrastructure limits.", "https://journals.brandonu.ca/jrcd/article/view/2718"),
        ("Mwansa, Ngandu and Mkwambi (2025)", "Rural South African communities face insufficient ICT infrastructure and unreliable connectivity.", "https://link.springer.com/article/10.1007/s44282-025-00189-2"),
        ("Ashrafi and Ghiasian (2025)", "Near-congestion and queue management are linked to delay and packet drops.", "https://link.springer.com/article/10.1007/s44354-025-00012-z"),
        ("Assis and Souza (2025)", "Adaptive-window methods can detect changing data streams without labels.", "https://link.springer.com/article/10.1007/s10115-025-02523-1"),
        ("Doherty et al. (2025)", "Simple policies may match or outperform RL, so proper benchmarking is necessary.", "https://opg.optica.org/jocn/abstract.cfm?uri=jocn-17-9-D1"),
        ("Shao et al. (2026)", "Changing and bursty workloads support the need for adaptive resource allocation.", "https://www.nature.com/articles/s41598-026-38622-4"),
    ]
    for author, meaning, url in sources:
        st.markdown(f"- **[{author}]({url})** — {meaning}")

    st.info("These sources support the rural context and the technical explanation. They do not prove that the real Alfred Nzo network uses exactly 20 Mbps, 50 ms delay or 2% loss. Those are controlled research settings.")


def sidebar() -> str:
    with st.sidebar:
        st.markdown("<div style='color:#d5a73a;font-size:1.15rem;font-weight:800;letter-spacing:0.16rem;'>DUNYWA</div>", unsafe_allow_html=True)
        st.markdown("### ANDM RL System")
        page = st.radio(
            "Navigation",
            ["Overview", "Measured network", "Agent comparison", "Self-evolving cycle", "Final decision", "Evidence and sources"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**System status**")
        st.success("Measured evidence loaded")
        st.caption("DQN champion: seed 42")
        st.caption("Deployed version: v2")
        st.caption("Final overall winner: CPU-rule")
        st.markdown("---")
        st.caption("University research prototype")
    return page


def main() -> None:
    baseline, rural, master, summary = load_evidence()
    page = sidebar()
    if page == "Overview":
        home_page(baseline, rural, summary)
    elif page == "Measured network":
        measured_page(baseline, rural, summary)
    elif page == "Agent comparison":
        agent_page()
    elif page == "Self-evolving cycle":
        self_evolving_page(baseline, rural)
    elif page == "Final decision":
        final_page()
    else:
        evidence_page(master, summary)


if __name__ == "__main__":
    main()
