import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data, get_kpis

st.set_page_config(
    page_title="Exploration — Adult Income",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #667eea; }
    .metric-label { font-size: 0.9rem; color: #666; margin-top: 0.2rem; }
    </style>
""",
    unsafe_allow_html=True,
)

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data("adult.csv")
n_rows, n_cols, missing_rate, high_income_rate = get_kpis(df)

st.markdown("# 📊 Exploration des Données")
st.markdown("Vue d'ensemble du dataset Adult Census Income et statistiques descriptives.")
st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("## 📈 Métriques Clés")
k1, k2, k3, k4 = st.columns(4)

kpis = [
    (f"{n_rows:,}", "Individus", "👥"),
    (f"{n_cols}", "Variables", "📋"),
    (f"{missing_rate:.1f}%", "Valeurs Manquantes", "❓"),
    (f"{high_income_rate:.1f}%", "Revenus >50K", "💰"),
]

for col, (val, label, icon) in zip([k1, k2, k3, k4], kpis):
    with col:
        st.markdown(
            f"""
<div class="metric-card">
  <div style="font-size:1.8rem;">{icon}</div>
  <div class="metric-value">{val}</div>
  <div class="metric-label">{label}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.divider()

# ── Aperçu ────────────────────────────────────────────────────────────────────
with st.expander("🔍 Aperçu du Dataset (100 premières lignes)", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Dimensions complètes : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

with st.expander("📋 Description des Colonnes", expanded=False):
    col_desc = {
        "age": "Âge de l'individu",
        "workclass": "Secteur d'activité (privé, public, indépendant…)",
        "fnlwgt": "Poids de pondération du recensement",
        "education": "Niveau d'études (texte)",
        "educational-num": "Niveau d'études (numérique)",
        "marital-status": "Statut marital",
        "occupation": "Profession",
        "relationship": "Relation familiale",
        "race": "Origine ethnique",
        "gender": "Genre (Male / Female)",
        "capital-gain": "Plus-values en capital",
        "capital-loss": "Moins-values en capital",
        "hours-per-week": "Heures travaillées par semaine",
        "native-country": "Pays de naissance",
        "income": "Revenu annuel (≤50K ou >50K) — variable cible",
    }
    desc_df = pd.DataFrame(
        [{"Colonne": k, "Type": str(df[k].dtype), "Description": v}
         for k, v in col_desc.items()]
    )
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

st.divider()

# ── Visualisations ────────────────────────────────────────────────────────────
st.markdown("## 📉 Visualisations")

# ── Viz 1 : Distribution de la variable cible ─────────────────────────────────
st.markdown("### 1️⃣ Distribution de la Variable Cible")
income_counts = df["income"].value_counts().reset_index()
income_counts.columns = ["Revenu", "Nombre"]

v1a, v1b = st.columns(2)
with v1a:
    fig = px.pie(
        income_counts,
        names="Revenu",
        values="Nombre",
        color_discrete_map={"<=50K": "#667eea", ">50K": "#764ba2"},
        title="Répartition des classes de revenus",
    )
    fig.update_traces(textinfo="percent+label", pull=[0, 0.05])
    st.plotly_chart(fig, use_container_width=True)

with v1b:
    fig2 = px.bar(
        income_counts,
        x="Revenu",
        y="Nombre",
        color="Revenu",
        color_discrete_map={"<=50K": "#667eea", ">50K": "#764ba2"},
        title="Nombre d'individus par classe",
        text="Nombre",
    )
    fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.info(
    f"⚠️ **Déséquilibre de classes** : {high_income_rate:.1f}% de la population gagne >50K. "
    "Ce déséquilibre (environ 75%/25%) peut biaiser les modèles ML vers la classe majoritaire."
)

st.divider()

# ── Viz 2 : Comparaison par attribut sensible ─────────────────────────────────
st.markdown("### 2️⃣ Revenus >50K par Attributs Sensibles")

tab1, tab2, tab3 = st.tabs(["⚧ Genre", "🌍 Race", "🎂 Âge"])

with tab1:
    gender_income = (
        df.groupby(["gender", "income"]).size().reset_index(name="count")
    )
    gender_pct = (
        df.groupby("gender")["income_binary"]
        .mean()
        .reset_index()
        .rename(columns={"income_binary": "taux_haut_revenu"})
    )
    gender_pct["taux_haut_revenu"] *= 100

    g1, g2 = st.columns(2)
    with g1:
        fig = px.bar(
            gender_income,
            x="gender",
            y="count",
            color="income",
            barmode="group",
            color_discrete_map={"<=50K": "#667eea", ">50K": "#764ba2"},
            title="Distribution des revenus par genre",
            labels={"gender": "Genre", "count": "Nombre", "income": "Revenu"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig2 = px.bar(
            gender_pct,
            x="gender",
            y="taux_haut_revenu",
            color="gender",
            color_discrete_map={"Male": "#667eea", "Female": "#e74c3c"},
            title="Taux de revenus >50K par genre (%)",
            text=gender_pct["taux_haut_revenu"].round(1).astype(str) + "%",
            labels={"gender": "Genre", "taux_haut_revenu": "Taux (%)"},
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    race_pct = (
        df.groupby("race")["income_binary"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "taux_haut_revenu", "count": "effectif"})
    )
    race_pct["taux_haut_revenu"] *= 100
    race_pct = race_pct.sort_values("taux_haut_revenu", ascending=False)

    r1_col, r2_col = st.columns(2)
    with r1_col:
        fig = px.bar(
            race_pct,
            x="race",
            y="taux_haut_revenu",
            color="taux_haut_revenu",
            color_continuous_scale="Purples",
            title="Taux de revenus >50K par race (%)",
            text=race_pct["taux_haut_revenu"].round(1).astype(str) + "%",
            labels={"race": "Race", "taux_haut_revenu": "Taux (%)"},
        )
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    with r2_col:
        fig2 = px.bar(
            race_pct,
            x="race",
            y="effectif",
            color="race",
            title="Effectifs par groupe racial",
            labels={"race": "Race", "effectif": "Nombre"},
        )
        fig2.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        right=False,
    )
    age_pct = (
        df.groupby("age_group", observed=False)["income_binary"]
        .mean()
        .reset_index()
        .rename(columns={"income_binary": "taux_haut_revenu"})
    )
    age_pct["taux_haut_revenu"] *= 100

    a1, a2 = st.columns(2)
    with a1:
        fig = px.bar(
            age_pct,
            x="age_group",
            y="taux_haut_revenu",
            color="taux_haut_revenu",
            color_continuous_scale="Viridis",
            title="Taux >50K par tranche d'âge (%)",
            text=age_pct["taux_haut_revenu"].round(1).astype(str) + "%",
            labels={"age_group": "Tranche d'âge", "taux_haut_revenu": "Taux (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        fig2 = px.histogram(
            df,
            x="age",
            color="income",
            nbins=40,
            color_discrete_map={"<=50K": "#667eea", ">50K": "#764ba2"},
            title="Distribution de l'âge par revenu",
            barmode="overlay",
            opacity=0.7,
            labels={"age": "Âge", "count": "Nombre"},
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Viz 3 : Scatter ───────────────────────────────────────────────────────────
st.markdown("### 3️⃣ Relation Âge / Heures Travaillées")

scatter_sample = df.sample(min(3000, len(df)), random_state=42)
fig = px.scatter(
    scatter_sample,
    x="age",
    y="hours-per-week",
    color="income",
    color_discrete_map={"<=50K": "#667eea", ">50K": "#764ba2"},
    opacity=0.5,
    title="Âge vs Heures travaillées (échantillon 3000)",
    labels={"age": "Âge", "hours-per-week": "Heures/semaine", "income": "Revenu"},
    hover_data=["occupation", "education"],
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Viz 4 : Heatmap corrélations ──────────────────────────────────────────────
st.markdown("### 4️⃣ Heatmap de Corrélations")

numeric_cols = ["age", "educational-num", "capital-gain", "capital-loss",
                "hours-per-week", "income_binary"]
corr_matrix = df[numeric_cols].corr().round(2)

fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 11},
    )
)
fig.update_layout(title="Matrice de corrélation des variables numériques", height=450)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Viz 5 : Box plot éducation ─────────────────────────────────────────────────
st.markdown("### 5️⃣ Distribution de l'Éducation par Revenu")

edu_order = df.groupby("education")["educational-num"].mean().sort_values().index.tolist()
fig = px.box(
    df,
    x="education",
    y="hours-per-week",
    color="income",
    color_discrete_map={"<=50K": "#667eea", ">50K": "#764ba2"},
    category_orders={"education": edu_order},
    title="Heures travaillées par niveau d'éducation et revenu",
    labels={"education": "Éducation", "hours-per-week": "Heures/semaine"},
)
fig.update_layout(xaxis_tickangle=-30, height=500)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Statistiques descriptives ─────────────────────────────────────────────────
st.markdown("### 📊 Statistiques Descriptives")
st.dataframe(
    df[["age", "educational-num", "hours-per-week", "capital-gain", "capital-loss"]]
    .describe()
    .round(2),
    use_container_width=True,
)
