import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data
from utils.fairness import demographic_parity_difference, disparate_impact_ratio

st.set_page_config(
    page_title="Modélisation — Adult Income",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .model-card {
        background: white; border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    .perf-val { font-size: 2rem; font-weight: 800; color: #667eea; }
    </style>
""",
    unsafe_allow_html=True,
)

df_raw = load_data("adult.csv")


@st.cache_data
def prepare_and_train(model_type="Logistic Regression"):
    df = df_raw.copy()
    # Encodage
    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "gender", "native-country"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = ["age", "workclass", "educational-num", "marital-status",
                    "occupation", "relationship", "race", "gender",
                    "capital-gain", "capital-loss", "hours-per-week"]
    X = df[feature_cols]
    y = df["income_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=300, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Rétablir les attributs sensibles sur le test set
    test_idx = X_test.index
    sensitive = {
        "gender_raw": df_raw.loc[test_idx, "gender"].values,
        "race_raw": df_raw.loc[test_idx, "race"].values,
        "age_raw": df_raw.loc[test_idx, "age"].values,
    }

    return model, X_test, y_test.values, y_pred, feature_cols, sensitive


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 Modélisation & Fairness sur Prédictions")
st.markdown(
    "Entraînement d'un modèle ML et évaluation des métriques de fairness **sur les prédictions**."
)
st.divider()

model_choice = st.selectbox(
    "Choisir le modèle :",
    ["Logistic Regression", "Random Forest"],
    index=0,
)

with st.spinner("Entraînement du modèle…"):
    model, X_test, y_test, y_pred, feature_cols, sensitive = prepare_and_train(model_choice)

st.success(f"✅ Modèle **{model_choice}** entraîné sur 80% des données (test : 20%)")
st.divider()

# ── Performances globales ─────────────────────────────────────────────────────
st.markdown("## 📊 Performances Globales")

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

p1, p2, p3, p4 = st.columns(4)
for col, label, val, icon in zip(
    [p1, p2, p3, p4],
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    [acc, prec, rec, f1],
    ["🎯", "🔍", "📡", "⚖️"],
):
    with col:
        color = "#27ae60" if val >= 0.8 else "#f39c12" if val >= 0.7 else "#e74c3c"
        st.markdown(
            f"""
<div class="model-card" style="text-align:center;">
  <div style="font-size:1.5rem;">{icon}</div>
  <div class="perf-val" style="color:{color};">{val:.3f}</div>
  <div style="color:#666; font-size:0.9rem;">{label}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.divider()

# ── Matrice de confusion globale ──────────────────────────────────────────────
st.markdown("## 🔲 Matrice de Confusion Globale")
cm = confusion_matrix(y_test, y_pred)
fig_cm = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Prédit ≤50K", "Prédit >50K"],
        y=["Réel ≤50K", "Réel >50K"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
    )
)
fig_cm.update_layout(title="Matrice de confusion (test set)", height=350)
st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# ── Fairness sur prédictions : Genre ─────────────────────────────────────────
st.markdown("## ⚖️ Métriques de Fairness sur les Prédictions du Modèle")
st.markdown(
    "On mesure ici si le modèle est **équitable** dans ses prédictions selon les attributs sensibles."
)

tab1, tab2, tab3 = st.tabs(["⚧ Genre", "🌍 Race", "🎂 Âge"])

with tab1:
    dpd_g = demographic_parity_difference(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_attribute=sensitive["gender_raw"],
    )
    di_g = disparate_impact_ratio(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_attribute=sensitive["gender_raw"],
        unprivileged_value="Female",
        privileged_value="Male",
    )

    gm1, gm2 = st.columns(2)
    with gm1:
        st.metric("Parité Démographique (diff.)", f"{dpd_g['difference']:.3f}",
                  help="Idéal : 0. Seuil toléré : < 0.1")
    with gm2:
        st.metric("Ratio d'Impact Disproportionné", f"{di_g['ratio']:.3f}",
                  help="Idéal : 1. Règle des 4/5 : ≥ 0.8")

    gender_results = pd.DataFrame(
        [{"Genre": k, "Taux prédit >50K (%)": v * 100}
         for k, v in dpd_g["rates"].items()]
    )
    fig_g = px.bar(
        gender_results, x="Genre", y="Taux prédit >50K (%)",
        color="Genre",
        color_discrete_map={"Male": "#667eea", "Female": "#e74c3c"},
        title=f"Taux prédit >50K par genre — {model_choice}",
        text=gender_results["Taux prédit >50K (%)"].round(1).astype(str) + "%",
    )
    fig_g.update_layout(showlegend=False)
    st.plotly_chart(fig_g, use_container_width=True)

    # Confusion matrices par genre
    st.markdown("**Matrices de Confusion par Genre**")
    cm_cols = st.columns(2)
    for i, gender in enumerate(["Male", "Female"]):
        mask = sensitive["gender_raw"] == gender
        if mask.sum() > 0:
            cm_g = confusion_matrix(y_test[mask], y_pred[mask])
            fig_cg = go.Figure(
                data=go.Heatmap(
                    z=cm_g,
                    x=["Prédit ≤50K", "Prédit >50K"],
                    y=["Réel ≤50K", "Réel >50K"],
                    colorscale="Blues",
                    text=cm_g,
                    texttemplate="%{text}",
                    textfont={"size": 14},
                )
            )
            acc_g = accuracy_score(y_test[mask], y_pred[mask])
            fig_cg.update_layout(
                title=f"{gender} — Accuracy : {acc_g:.3f}", height=300
            )
            with cm_cols[i]:
                st.plotly_chart(fig_cg, use_container_width=True)

with tab2:
    dpd_r = demographic_parity_difference(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_attribute=sensitive["race_raw"],
    )

    races = list(dpd_r["rates"].keys())
    min_r = min(dpd_r["rates"], key=dpd_r["rates"].get)
    max_r = max(dpd_r["rates"], key=dpd_r["rates"].get)

    di_r2 = disparate_impact_ratio(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_attribute=sensitive["race_raw"],
        unprivileged_value=min_r,
        privileged_value=max_r,
    )

    rm1, rm2 = st.columns(2)
    with rm1:
        st.metric("Parité Démographique (diff.)", f"{dpd_r['difference']:.3f}")
    with rm2:
        st.metric(f"Ratio DI ({min_r} / {max_r})", f"{di_r2['ratio']:.3f}")

    race_results = pd.DataFrame(
        [{"Race": k, "Taux prédit >50K (%)": v * 100}
         for k, v in dpd_r["rates"].items()]
    ).sort_values("Taux prédit >50K (%)", ascending=False)

    fig_r = px.bar(
        race_results,
        x="Race",
        y="Taux prédit >50K (%)",
        color="Taux prédit >50K (%)",
        color_continuous_scale="RdYlGn",
        title=f"Taux prédit >50K par race — {model_choice}",
        text=race_results["Taux prédit >50K (%)"].round(1).astype(str) + "%",
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Accuracy par race
    acc_by_race = []
    for race in races:
        mask = sensitive["race_raw"] == race
        if mask.sum() > 0:
            acc_by_race.append({
                "Race": race,
                "Accuracy": accuracy_score(y_test[mask], y_pred[mask]),
                "Effectif": mask.sum(),
            })
    acc_race_df = pd.DataFrame(acc_by_race).sort_values("Accuracy", ascending=False)
    fig_acc = px.bar(
        acc_race_df,
        x="Race",
        y="Accuracy",
        color="Accuracy",
        color_continuous_scale="Blues",
        title="Accuracy par groupe racial",
        text=acc_race_df["Accuracy"].round(3).astype(str),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with tab3:
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_groups = pd.cut(
        sensitive["age_raw"], bins=age_bins, labels=age_labels, right=False
    )

    age_results = []
    for g in age_labels:
        mask = age_groups == g
        if mask.sum() > 0:
            age_results.append({
                "Tranche": g,
                "Taux prédit >50K (%)": y_pred[mask].mean() * 100,
                "Taux réel >50K (%)": y_test[mask].mean() * 100,
                "Accuracy": accuracy_score(y_test[mask], y_pred[mask]),
                "Effectif": mask.sum(),
            })
    age_df = pd.DataFrame(age_results)

    at1, at2 = st.columns(2)
    with at1:
        fig_a1 = px.bar(
            age_df,
            x="Tranche",
            y="Taux prédit >50K (%)",
            color="Taux prédit >50K (%)",
            color_continuous_scale="RdYlGn",
            title="Taux prédit >50K par tranche d'âge",
            text=age_df["Taux prédit >50K (%)"].round(1).astype(str) + "%",
        )
        st.plotly_chart(fig_a1, use_container_width=True)
    with at2:
        fig_a2 = px.bar(
            age_df,
            x="Tranche",
            y="Accuracy",
            color="Accuracy",
            color_continuous_scale="Blues",
            title="Accuracy par tranche d'âge",
            text=age_df["Accuracy"].round(3).astype(str),
        )
        st.plotly_chart(fig_a2, use_container_width=True)

    # Comparaison réel vs prédit
    fig_a3 = go.Figure()
    fig_a3.add_trace(go.Bar(
        name="Taux réel >50K (%)",
        x=age_df["Tranche"],
        y=age_df["Taux réel >50K (%)"],
        marker_color="#667eea",
    ))
    fig_a3.add_trace(go.Bar(
        name="Taux prédit >50K (%)",
        x=age_df["Tranche"],
        y=age_df["Taux prédit >50K (%)"],
        marker_color="#764ba2",
    ))
    fig_a3.update_layout(
        barmode="group",
        title="Comparaison taux réel vs taux prédit par tranche d'âge",
    )
    st.plotly_chart(fig_a3, use_container_width=True)

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
if model_choice == "Random Forest":
    st.markdown("## 🌲 Importance des Variables (Random Forest)")
    fi = pd.DataFrame({
        "Variable": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi,
        x="Importance",
        y="Variable",
        orientation="h",
        color="Importance",
        color_continuous_scale="Purples",
        title="Importance des variables — Random Forest",
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.info(
        "💡 Les variables **age**, **capital-gain** et **educational-num** sont typiquement "
        "les plus influentes. Notez que **race** et **gender** (encodés) apparaissent aussi, "
        "ce qui confirme que le modèle utilise ces attributs sensibles dans ses prédictions."
    )

st.divider()
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Modèle entraîné sur le dataset UCI Adult Census Income (1994) — À titre académique uniquement"
    "</div>",
    unsafe_allow_html=True,
)
