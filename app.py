import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI SOC Dashboard Pro",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# LIVE AUTO REFRESH
# -----------------------------
REFRESH_INTERVAL = 5
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.rerun()

# -----------------------------
# CSS (DARK + BLINK)
# -----------------------------
st.markdown("""
<style>
.main { background-color: #0e1117; }
@keyframes blink {
    0% { background-color: #8B0000; }
    50% { background-color: #FF0000; }
    100% { background-color: #8B0000; }
}
.blink-critical {
    animation: blink 1s infinite;
    color: white;
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("ML-Powered Network Intrusion Detection System")
st.caption("Anomaly-Based Detection using Isolation Forest | SOC Simulation")

# -----------------------------
# DATA LOADING
# -----------------------------
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Network Traffic CSV", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        n = 1200
        df = pd.DataFrame({
            "timestamp": pd.date_range(datetime.now(), periods=n, freq="s"),
            "src_ip": np.random.choice(
                ["10.0.0.1", "192.168.1.45", "172.16.0.5", "203.0.113.9"], n),
            "dst_ip": np.random.choice(
                ["8.8.8.8", "1.1.1.1", "10.0.0.254"], n),
            "dst_port": np.random.choice([22, 80, 443, 3389, 8080], n),
            "packet_rate": np.random.randint(100, 5000, n),
            "byte_rate": np.random.randint(1000, 1000000, n),
            "tcp_syn": np.random.randint(0, 20, n),
            "tcp_ack": np.random.randint(0, 20, n),
            "tcp_rst": np.random.randint(0, 10, n),
            "failed_connections": np.random.randint(0, 15, n)
        })

        df["Actual_Label"] = np.where(
            (df["packet_rate"] > 4400) | (df["failed_connections"] > 12), 1, 0
        )
    return df

data = load_data(uploaded_file)

# -----------------------------
# ML ENGINE
# -----------------------------
st.sidebar.header("ML Settings")
contamination = st.sidebar.slider("Anomaly Sensitivity", 0.01, 0.20, 0.05)

features = [
    "packet_rate", "byte_rate",
    "tcp_syn", "tcp_ack", "tcp_rst",
    "failed_connections"
]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

model = IsolationForest(contamination=contamination, random_state=42)
data["ml_pred"] = model.fit_predict(scaled_data)
data["AI_Flagged"] = data["ml_pred"].map({1: 0, -1: 1})
data["Anomaly_Score"] = model.decision_function(scaled_data)

# -----------------------------
# SEVERITY ENGINE
# -----------------------------
def get_severity(row):
    if row["AI_Flagged"] == 1 and row["Anomaly_Score"] < -0.15:
        return "Critical"
    elif row["AI_Flagged"] == 1:
        return "High"
    elif row["packet_rate"] > 3000:
        return "Medium"
    return "Low"

data["Severity"] = data.apply(get_severity, axis=1)

severity_colors = {
    "Low": "#43FA8F",
    "Medium": "#FFD634",
    "High": "#FF871D",
    "Critical": "#FA2209"
}

# -----------------------------
# KPI DASHBOARD
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Flows", len(data))
c2.metric("AI Alerts", int(data["AI_Flagged"].sum()))
c3.metric("Precision", f"{precision_score(data['Actual_Label'], data['AI_Flagged']):.2f}")
c4.metric("Engine Status", "LIVE")

if (data["Severity"] == "Critical").sum() > 0:
    st.markdown(
        '<div class="blink-critical">CRITICAL ANOMALIES DETECTED </div>',
        unsafe_allow_html=True
    )

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Threat Stream",
    "Anomaly Analysis",
    "Alert Logs",
    "Model Metrics"
])

# -----------------------------
# TAB 1: LIVE STREAM
# -----------------------------
with tab1:
    fig_live = px.scatter(
        data.tail(300),
        x="timestamp",
        y="packet_rate",
        color="Severity",
        size="failed_connections",
        color_discrete_map=severity_colors,
        template="plotly_dark",
        title="Live Network Traffic Stream"
    )
    st.plotly_chart(fig_live, use_container_width=True)

# -----------------------------
# TAB 2: ATTACKERS + SCORE + FILTER
# -----------------------------
with tab2:
    st.subheader("Top 5 Attacker IPs")

    attackers = (
        data[data["AI_Flagged"] == 1]["src_ip"]
        .value_counts()
        .reset_index()
        .head(5)
    )
    attackers.columns = ["src_ip", "count"]

    fig_attack = px.bar(
        attackers,
        x="count",
        y="src_ip",
        orientation="h",
        color="count",
        color_continuous_scale="Reds",
        template="plotly_dark",
        title="Most Frequent Sources of Anomalies"
    )
    st.plotly_chart(fig_attack, use_container_width=True)

    attacker_options = ["ALL"] + attackers["src_ip"].tolist()
    selected_ip = st.selectbox("Focus on Attacker IP", attacker_options)

    if selected_ip == "ALL":
        filtered_data = data
    else:
        filtered_data = data[data["src_ip"] == selected_ip]

    st.divider()
    st.subheader("Anomaly Score Distribution")

    fig_score = px.histogram(
        filtered_data,
        x="Anomaly_Score",
        color="AI_Flagged",
        nbins=50,
        template="plotly_dark",
        title="Isolation Forest Scores (Lower = More Suspicious)"
    )

    fig_score.add_vline(
        x=-0.15,
        line_dash="dash",
        line_color="red",
        annotation_text="Critical Threshold"
    )

    st.plotly_chart(fig_score, use_container_width=True)

# -----------------------------
# TAB 3: ALERT LOGS
# -----------------------------
with tab3:
    st.dataframe(
        filtered_data[filtered_data["AI_Flagged"] == 1]
        .sort_values("Anomaly_Score")[[
            "timestamp", "src_ip", "dst_ip",
            "dst_port", "packet_rate",
            "failed_connections",
            "Anomaly_Score", "Severity"
        ]],
        use_container_width=True
    )

# -----------------------------
# TAB 4: MODEL METRICS
# -----------------------------
with tab4:
    cm = confusion_matrix(data["Actual_Label"], data["AI_Flagged"])
    fig_cm = ff.create_annotated_heatmap(
        cm,
        x=["Pred Normal", "Pred Anomaly"],
        y=["Actual Normal", "Actual Anomaly"],
        colorscale="Purples"
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.code(classification_report(data["Actual_Label"], data["AI_Flagged"]))

# -----------------------------
# EXPORT
# -----------------------------
st.sidebar.divider()
st.sidebar.download_button(
    "📥 Export SOC Audit Logs",
    data.to_csv(index=False).encode("utf-8"),
    "ai_soc_final_report.csv",
    "text/csv"
)

st.success(f"🔄 SOC Engine Updated @ {datetime.now().strftime('%H:%M:%S')}")
