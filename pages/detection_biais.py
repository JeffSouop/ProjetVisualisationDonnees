import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data
from utils.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    compute_age_group_fairness,
)

st.set_page_config(
    page_title="Détection de Biais — Adult Income",
    page_icon="⚠️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .bias-card {
        background: white; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .biased { border-left: 5px solid #e74c3c; }
    .fair   { border-left: 5px solid #27ae60; }
    .warn   { border-left: 5px solid #f39c12; }
    .metric-big { font-size: 2.5rem; font-weight: 800; }
    .red  { color: #e74c3c; }
    .green{ color: #27ae60; }
    .orange { color: #f39c12; }
    </style>
""",
    unsafe_allow_html=True,
)

df = load_data("adult.csv")
df["income_binary"] = (df["income"] == ">50K").astype(int)

st.markdown("# ⚠️ Détection de Biais")
st.markdown(
    "Analyse des disparités de revenus selon le **genre**, la **race** et l'**âge**."
)
st.divider()

# ────────────────────────────────────────────────────────────────────
# SECTION 1 : BIAIS DE GENRE
# ────────────────────────────────────────────────────────────────────
st.markdown("## ⚧ Analyse du Biais de Genre")

with st.expander("ℹ️ Pourquoi ce biais est-il problématique ?", expanded=True):
    st.markdown(
        """
**Attribut sensible : Genre (Male / Female)**

Le genre est un attribut protégé dans la plupart des réglementations anti-discrimination.  
Dans le recensement de 1994, les femmes étaient encore massivement sous-représentées dans les 
postes à hauts salaires — non pas nécessairement par manque de compétences, mais à cause de 
**barrières systémiques** (plafond de verre, interruptions de carrière, ségrégation 
sectorielle). 

Si un modèle ML apprend sur ces données, il risque d'associer le fait d'être une femme à une 
plus faible probabilité de gagner >50K, **reproduisant ainsi une discrimination historique** 
dans des décisions futures (crédit, recrutement, etc.).
"""
    )

# Métriques
dpd_gender = demographic_parity_difference(
    y_true=df["income_binary"].values,
    y_pred=df["income_binary"].values,
    sensitive_attribute=df["gender"].values,
)
di_gender = disparate_impact_ratio(
    y_true=df["income_binary"].values,
    y_pred=df["income_binary"].values,
    sensitive_attribute=df["gender"].values,
    unprivileged_value="Female",
    privileged_value="Male",
)

m1, m2, m3 = st.columns(3)
with m1:
    dpd_val = dpd_gender["difference"]
    color = "red" if dpd_val > 0.1 else "orange" if dpd_val > 0.05 else "green"
    st.markdown(
        f"""
<div class="bias-card biased">
  <div style="font-size:0.9rem; color:#666; font-weight:600;">
    📏 Différence de Parité Démographique
  </div>
  <div class="metric-big {color}">{dpd_val:.3f}</div>
  <div style="font-size:0.85rem; color:#888; margin-top:0.3rem;">
    Idéal : 0.000 &nbsp;|&nbsp; Seuil toléré : &lt; 0.1
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with m2:
    di_val = di_gender["ratio"]
    color = "red" if di_val < 0.8 else "orange" if di_val < 0.9 else "green"
    st.markdown(
        f"""
<div class="bias-card {'biased' if di_val < 0.8 else 'warn'}">
  <div style="font-size:0.9rem; color:#666; font-weight:600;">
    ⚖️ Ratio d'Impact Disproportionné (DI)
  </div>
  <div class="metric-big {color}">{di_val:.3f}</div>
  <div style="font-size:0.85rem; color:#888; margin-top:0.3rem;">
    Idéal : 1.000 &nbsp;|&nbsp; Règle 4/5 : ≥ 0.800
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with m3:
    rate_m = dpd_gender["rates"].get("Male", 0) * 100
    rate_f = dpd_gender["rates"].get("Female", 0) * 100
    st.markdown(
        f"""
<div class="bias-card warn">
  <div style="font-size:0.9rem; color:#666; font-weight:600;">
    📊 Taux >50K par Genre
  </div>
  <div style="margin-top:0.5rem;">
    <b>Hommes :</b> <span class="metric-big" style="font-size:1.8rem; color:#667eea;">
      {rate_m:.1f}%</span><br>
    <b>Femmes :</b> <span class="metric-big" style="font-size:1.8rem; color:#e74c3c;">
      {rate_f:.1f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# Visualisation genre
rates_gender_df = pd.DataFrame(
    [{"Genre": k, "Taux >50K (%)": v * 100} for k, v in dpd_gender["rates"].items()]
)
fig = px.bar(
    rates_gender_df,
    x="Genre",
    y="Taux >50K (%)",
    color="Genre",
    color_discrete_map={"Male": "#667eea", "Female": "#e74c3c"},
    title="Taux de revenus >50K par genre",
    text=rates_gender_df["Taux >50K (%)"].round(1).astype(str) + "%",
    height=350,
)
fig.add_hline(
    y=df["income_binary"].mean() * 100,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"Moyenne globale : {df['income_binary'].mean()*100:.1f}%",
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
**🔎 Interprétation du biais de genre :**
- La différence de parité démographique de **{:.3f}** signifie que les hommes ont {:.1f} points 
  de pourcentage de plus de chances d'être classés >50K.
- Le ratio DI de **{:.3f}** est {} la règle des 4/5 (0.8), ce qui {} une discrimination 
  légale significative selon les standards EEOC.
- En pratique, une femme ayant exactement le même profil qu'un homme aurait une probabilité 
  bien plus faible d'obtenir un crédit ou un poste si ce modèle était utilisé.
- **Recommandation** : Appliquer un rééquilibrage (re-sampling), un prétraitement adversarial, 
  ou des contraintes de fairness lors de l'entraînement.
""".format(
        dpd_val,
        dpd_val * 100,
        di_val,
        "en-dessous de" if di_val < 0.8 else "au-dessus de",
        "indique" if di_val < 0.8 else "n'indique pas encore formellement",
    )
)

st.divider()

# ────────────────────────────────────────────────────────────────────
# SECTION 2 : BIAIS DE RACE
# ────────────────────────────────────────────────────────────────────
st.markdown("## 🌍 Analyse du Biais de Race")

with st.expander("ℹ️ Pourquoi ce biais est-il problématique ?", expanded=True):
    st.markdown(
        """
**Attribut sensible : Race**

La race est un attribut protégé légalement. Les inégalités raciales dans le système économique 
américain sont documentées depuis des décennies. Dans ce dataset, les groupes minoritaires 
(Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other) apparaissent moins souvent dans la 
tranche >50K — non seulement à cause de différences de formation ou d'accès aux opportunités, 
mais aussi à cause de **discriminations directes et indirectes** sur le marché du travail.

Un modèle ML qui apprend ces patterns va **amplifier** ces inégalités en production.
"""
    )

dpd_race = demographic_parity_difference(
    y_true=df["income_binary"].values,
    y_pred=df["income_binary"].values,
    sensitive_attribute=df["race"].values,
)

# Choisir le groupe le moins favorisé
rates_race = {k: v * 100 for k, v in dpd_race["rates"].items()}
min_race = min(rates_race, key=rates_race.get)
max_race = max(rates_race, key=rates_race.get)

di_race = disparate_impact_ratio(
    y_true=df["income_binary"].values,
    y_pred=df["income_binary"].values,
    sensitive_attribute=df["race"].values,
    unprivileged_value=min_race,
    privileged_value=max_race,
)

r1_col, r2_col = st.columns(2)
with r1_col:
    dpd_r = dpd_race["difference"]
    color = "red" if dpd_r > 0.1 else "orange" if dpd_r > 0.05 else "green"
    st.markdown(
        f"""
<div class="bias-card biased">
  <div style="font-size:0.9rem; color:#666; font-weight:600;">
    📏 Différence de Parité Démographique (Race)
  </div>
  <div class="metric-big {color}">{dpd_r:.3f}</div>
  <div style="font-size:0.85rem; color:#888; margin-top:0.3rem;">
    Entre <b>{max_race}</b> ({rates_race[max_race]:.1f}%) 
    et <b>{min_race}</b> ({rates_race[min_race]:.1f}%)
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with r2_col:
    di_r = di_race["ratio"]
    color = "red" if di_r < 0.8 else "orange" if di_r < 0.9 else "green"
    st.markdown(
        f"""
<div class="bias-card {'biased' if di_r < 0.8 else 'warn'}">
  <div style="font-size:0.9rem; color:#666; font-weight:600;">
    ⚖️ Ratio DI ({min_race} vs {max_race})
  </div>
  <div class="metric-big {color}">{di_r:.3f}</div>
  <div style="font-size:0.85rem; color:#888; margin-top:0.3rem;">
    Règle des 4/5 : ≥ 0.800 &nbsp;|&nbsp; Idéal : 1.000
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# Bar chart race
rates_race_df = pd.DataFrame(
    [{"Race": k, "Taux >50K (%)": v} for k, v in rates_race.items()]
).sort_values("Taux >50K (%)", ascending=False)

fig = px.bar(
    rates_race_df,
    x="Race",
    y="Taux >50K (%)",
    color="Taux >50K (%)",
    color_continuous_scale="RdYlGn",
    title="Taux de revenus >50K par groupe racial",
    text=rates_race_df["Taux >50K (%)"].round(1).astype(str) + "%",
    height=380,
)
fig.add_hline(
    y=df["income_binary"].mean() * 100,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"Moyenne : {df['income_binary'].mean()*100:.1f}%",
)
st.plotly_chart(fig, use_container_width=True)

# Effectifs par race + croisement
race_income = (
    df.groupby(["race", "income"]).size().reset_index(name="count")
)
fig2 = px.bar(
    race_income,
    x="race",
    y="count",
    color="income",
    barmode="stack",
    color_discrete_map={"<=50K": "#667eea", ">50K": "#764ba2"},
    title="Répartition des revenus par race (valeurs absolues)",
    labels={"race": "Race", "count": "Nombre"},
)
fig2.update_layout(xaxis_tickangle=-15)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    f"""
**🔎 Interprétation du biais de race :**
- Il existe un écart de **{dpd_r*100:.1f} points de pourcentage** entre le groupe le plus 
  favorisé ({max_race} : {rates_race[max_race]:.1f}%) et le moins favorisé 
  ({min_race} : {rates_race[min_race]:.1f}%).
- Le ratio DI de **{di_r:.3f}** indique que le groupe *{min_race}* a environ 
  **{di_r*100:.0f}%** de la probabilité** du groupe *{max_race}* d'obtenir une prédiction 
  >50K.
- Ce biais est partiellement expliqué par des inégalités d'accès à l'éducation et au marché du 
  travail, mais aussi par des discriminations directes capturées dans les données.
- **Recommandation** : Utiliser des algorithmes de fairness-aware (ex. : Fairlearn, AIF360) 
  ou corriger la représentation dans les données d'entraînement.
"""
)

st.divider()

# ────────────────────────────────────────────────────────────────────
# SECTION 3 : BIAIS D'ÂGE
# ────────────────────────────────────────────────────────────────────
st.markdown("## 🎂 Analyse du Biais d'Âge")

with st.expander("ℹ️ Pourquoi ce biais est-il problématique ?", expanded=True):
    st.markdown(
        """
**Attribut sensible : Âge**

Bien que l'âge soit corrélé à l'expérience et donc au salaire, certaines tranches d'âge sont 
désavantagées de manière disproportionnée : les **jeunes** manquent d'expérience par définition, 
et les **seniors** peuvent faire face à de l'âgisme sur le marché du travail. Un modèle qui 
pénalise l'âge risque de prendre des décisions discriminatoires vis-à-vis de ces groupes.
"""
    )

age_fairness = compute_age_group_fairness(
    df, target_col="income_binary", age_col="age"
)
age_fairness["Taux >50K (%)"] = age_fairness["positive_rate"] * 100

max_age = age_fairness.loc[age_fairness["positive_rate"].idxmax(), "age_group"]
min_age = age_fairness.loc[age_fairness["positive_rate"].idxmin(), "age_group"]
dpd_age = (
    age_fairness["positive_rate"].max() - age_fairness["positive_rate"].min()
)

a1, a2 = st.columns(2)
with a1:
    color = "red" if dpd_age > 0.2 else "orange" if dpd_age > 0.1 else "green"
    st.markdown(
        f"""
<div class="bias-card {'biased' if dpd_age > 0.2 else 'warn'}">
  <div style="font-size:0.9rem; color:#666; font-weight:600;">
    📏 Écart max entre tranches d'âge
  </div>
  <div class="metric-big {color}">{dpd_age:.3f}</div>
  <div style="font-size:0.85rem; color:#888; margin-top:0.3rem;">
    Tranche la plus haute : <b>{max_age}</b> | 
    La plus basse : <b>{min_age}</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with a2:
    youngest_rate = age_fairness[age_fairness["age_group"] == "<25"]["Taux >50K (%)"].values
    oldest_rate = age_fairness[age_fairness["age_group"] == "65+"]["Taux >50K (%)"].values
    y_val = youngest_rate[0] if len(youngest_rate) > 0 else 0
    o_val = oldest_rate[0] if len(oldest_rate) > 0 else 0
    st.markdown(
        f"""
<div class="bias-card warn">
  <div style="font-size:0.9rem; color:#666; font-weight:600;">
    📊 Groupes extrêmes
  </div>
  <div style="margin-top:0.5rem;">
    <b>Jeunes (&lt;25) :</b> <span style="font-size:1.5rem; font-weight:700; color:#e74c3c;">
      {y_val:.1f}%</span><br>
    <b>Seniors (65+) :</b> <span style="font-size:1.5rem; font-weight:700; color:#f39c12;">
      {o_val:.1f}%</span><br>
    <b>Pic (35-54) :</b> <span style="font-size:1.5rem; font-weight:700; color:#27ae60;">
      ~{age_fairness[age_fairness['age_group'].isin(['35-44','45-54'])]['Taux >50K (%)'].mean():.1f}%
    </span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

fig = px.bar(
    age_fairness,
    x="age_group",
    y="Taux >50K (%)",
    color="Taux >50K (%)",
    color_continuous_scale="RdYlGn",
    title="Taux de revenus >50K par tranche d'âge",
    text=age_fairness["Taux >50K (%)"].round(1).astype(str) + "%",
    labels={"age_group": "Tranche d'âge"},
    height=380,
)
fig.add_hline(
    y=df["income_binary"].mean() * 100,
    line_dash="dash",
    line_color="gray",
    annotation_text="Moyenne globale",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
**🔎 Interprétation du biais d'âge :**
- L'écart entre la tranche la plus favorisée (**{max_age}**) et la moins favorisée 
  (**{min_age}**) est de **{dpd_age*100:.1f} points de pourcentage**, ce qui est très 
  significatif.
- Les jeunes de moins de 25 ans ({y_val:.1f}%) et les seniors de 65+ ({o_val:.1f}%) sont 
  clairement défavorisés par rapport aux 35-54 ans qui représentent la tranche la plus productive.
- Pour les jeunes, ce biais est partiellement justifié par le manque d'expérience. Pour les 
  seniors, il peut masquer de l'âgisme dans le marché du travail de 1994.
- **Recommandation** : Neutraliser l'âge comme feature directe et introduire uniquement des 
  proxy d'expérience pertinents, ou appliquer des contraintes de fairness par tranche d'âge.
"""
)

st.divider()

# ────────────────────────────────────────────────────────────────────
# SYNTHÈSE
# ────────────────────────────────────────────────────────────────────
st.markdown("## 📋 Synthèse des Métriques de Fairness")

summary_data = {
    "Attribut Sensible": ["Genre", "Genre", "Race", "Race", "Âge"],
    "Métrique": [
        "Parité Démographique (diff.)",
        "Impact Disproportionné (ratio)",
        "Parité Démographique (diff.)",
        "Impact Disproportionné (ratio)",
        "Écart max tranches d'âge",
    ],
    "Valeur": [
        round(dpd_gender["difference"], 3),
        round(di_gender["ratio"], 3),
        round(dpd_race["difference"], 3),
        round(di_race["ratio"], 3),
        round(dpd_age, 3),
    ],
    "Seuil Critique": ["< 0.1", "≥ 0.8", "< 0.1", "≥ 0.8", "< 0.1"],
    "Statut": [
        "🔴 Biaisé" if dpd_gender["difference"] > 0.1 else "🟡 Attention",
        "🔴 Biaisé" if di_gender["ratio"] < 0.8 else "🟡 Attention",
        "🔴 Biaisé" if dpd_race["difference"] > 0.1 else "🟡 Attention",
        "🔴 Biaisé" if di_race["ratio"] < 0.8 else "🟡 Attention",
        "🔴 Biaisé" if dpd_age > 0.2 else "🟡 Attention",
    ],
}
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.error(
    "⚠️ **Conclusion globale** : Le dataset Adult Income présente des biais significatifs sur "
    "les trois attributs sensibles analysés. Tout modèle entraîné sans correction risque de "
    "reproduire et d'amplifier ces discriminations dans des contextes réels (crédit, "
    "recrutement, accès aux services)."
)
