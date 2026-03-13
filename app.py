import streamlit as st

st.set_page_config(
    page_title="Adult Income — Détection de Biais",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .hero-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .hero-subtitle { font-size: 1.2rem; color: #666; margin-top: 0.5rem; }
    .card {
        background: grey; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea; margin-bottom: 1rem;
    }
    .card h3 { color: #1a1a2e; margin-top: 0; }
    .badge {
        display: inline-block; background: #667eea; color: white;
        border-radius: 20px; padding: 2px 12px; font-size: 0.85rem;
        margin: 2px;
    }
    .warning-box {
        background: #fff3cd; border-left: 4px solid #ffc107;
        border-radius: 8px; padding: 1rem; margin: 1rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center; padding: 2rem 0 1rem 0;">
  <div class="hero-title">Adult Income — Détection de Biais</div>
  <div class="hero-subtitle">
    Analyse d'équité algorithmique sur le dataset UCI Adult Income
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# ── Contexte ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## Contexte & Problématique")
    st.markdown(
        """
Le **dataset Adult Income** (également appelé *Census Income*) est issu du recensement américain 
de 1994. Il contient des informations socio-démographiques sur **48 842 individus** et est 
couramment utilisé pour entraîner des modèles de machine learning qui prédisent si une personne 
gagne **plus ou moins de 50 000 $/an**.

Ce type de modèle peut être utilisé dans des contextes à fort enjeu : attribution de crédit, 
recrutement, ou accès à des prestations sociales. Or, si les données d'entraînement reflètent 
des inégalités historiques (par exemple, moins de femmes aux postes bien rémunérés en 1994), 
le modèle risque de **perpétuer et d'amplifier ces discriminations**.

Cette application explore et quantifie les biais présents dans ce dataset selon trois attributs 
sensibles : le **genre**, la **race** et l'**âge**. Elle répond à la question : *un modèle 
entraîné sur ces données serait-il équitable pour tous les groupes ?*
"""
    )

with col2:
    st.markdown("## Informations Dataset")
    st.markdown(
        """
<div class="card">
  <h3>Adult Census Income</h3>
  <p><b>Source :</b> UCI Machine Learning Repository</p>
  <p><b>Année :</b> 1994</p>
  <p><b>Taille :</b> 48 842 lignes × 15 colonnes</p>
  <p><b>Variable cible :</b> <code>income</code> (≤50K / >50K)</p>
  <p><b>Biais analysés :</b></p>
  <span class="badge">Genre</span>
  <span class="badge">Race</span>
  <span class="badge">Âge</span>
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()

# ── Navigation cards ──────────────────────────────────────────────────────────
st.markdown("## Navigation")

c1, c2, c3, c4 = st.columns(4)

pages_info = [
    ("🏠", "Accueil", "Présentation du projet et contexte", "#667eea"),
    ("📊", "Exploration", "KPIs, visualisations et statistiques descriptives", "#764ba2"),
    ("⚠️", "Détection de Biais", "Métriques de fairness et interprétation", "#e74c3c"),
    ("🤖", "Modélisation", "Entraînement ML et comparaison par groupe", "#27ae60"),
]

for col, (icon, name, desc, color) in zip([c1, c2, c3, c4], pages_info):
    with col:
        st.markdown(
            f"""
<div style="background:white; border-radius:12px; padding:1.2rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08);
            border-top: 4px solid {color}; text-align:center;">
  <div style="font-size:2rem;">{icon}</div>
  <div style="font-weight:700; color:#1a1a2e; margin:0.3rem 0;">{name}</div>
  <div style="font-size:0.85rem; color:#666;">{desc}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.divider()

# ── Variables sensibles ───────────────────────────────────────────────────────
st.markdown("## Attributs Sensibles Analysés")

r1, r2, r3 = st.columns(3)

with r1:
    st.markdown(
        """
<div class="card">
  <h3>⚧ Genre</h3>
  <p>En 1994, les femmes étaient largement sous-représentées dans les emplois bien rémunérés. 
  Un modèle entraîné sur ces données risque de défavoriser les femmes, même à profil 
  équivalent.</p>
  <span class="badge">Parité Démographique</span>
  <span class="badge">Impact Disproportionné</span>
</div>
""",
        unsafe_allow_html=True,
    )

with r2:
    st.markdown(
        """
<div class="card" style="border-left-color:#e74c3c;">
  <h3>Race</h3>
  <p>Les inégalités raciales systémiques se reflètent dans les données salariales. 
  Les groupes minoritaires apparaissent moins fréquemment dans la tranche >50K, 
  ce qui biaise les prédictions.</p>
  <span class="badge" style="background:#e74c3c;">Parité Démographique</span>
  <span class="badge" style="background:#e74c3c;">Ratio DI</span>
</div>
""",
        unsafe_allow_html=True,
    )

with r3:
    st.markdown(
        """
<div class="card" style="border-left-color:#27ae60;">
  <h3>Âge</h3>
  <p>L'âge est corrélé au revenu mais peut aussi masquer des discriminations indirectes. 
  Les jeunes et les seniors sont sous-représentés dans les hauts revenus pour des 
  raisons qui ne reflètent pas toujours le mérite.</p>
  <span class="badge" style="background:#27ae60;">Tranches d'Âge</span>
  <span class="badge" style="background:#27ae60;">Parité par Groupe</span>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Dataset : UCI Adult Census Income (1994) | "
    "Projet Streamlit — Parcours A : Détection de Biais"
    "</div>",
    unsafe_allow_html=True,
)
