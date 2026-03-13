# Adult Income — Détection de Biais

Application Streamlit d'analyse d'équité algorithmique sur le dataset **UCI Adult Census Income**.

## Objectif

Détecter et quantifier les biais présents dans les données et les prédictions d'un modèle ML selon trois attributs sensibles : **Genre**, **Race**, **Âge**.

## Dataset

- **Source** : [UCI Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset/data)
- **Taille** : 48 842 individus × 15 variables
- **Variable cible** : `income` (≤50K$ / >50K$ par an)
- **Contexte** : Recensement américain de 1994

## Structure du Projet

```
adult_income_app/
│
├── app.py                          # Page d'accueil
├── adult.csv                       # Dataset
├── requirements.txt
│
├── pages/
│   ├── exploration.py         # Exploration des données (KPIs + 5 viz)
│   ├── détection_biais.py     # Métriques de fairness
│   └── modélisation.py        # Modèle ML + fairness sur prédictions
│
└── utils/
    ├── __init__.py
    ├── data_loader.py               # Chargement et cache des données
    └── fairness.py                  # Métriques : parité, DI, égalité des chances
```

## Pages de l'Application

| Page | Contenu |
|------|---------|
| **Accueil** | Contexte, problématique, attributs sensibles |
| **Exploration** | 4 KPIs, aperçu interactif, 5 visualisations |
| **Détection de Biais** | Parité démographique, DI ratio, analyse genre/race/âge |
| **Modélisation** | Régression Logistique / Random Forest + matrices de confusion par groupe |

## Installation Locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement sur Streamlit Cloud

1. Créer un repository GitHub et y pousser ce projet
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter votre GitHub
4. Sélectionner le repo → Main file path : `app.py`
5. Cliquer sur **Deploy**

## 📐 Métriques de Fairness Implémentées

- **Différence de Parité Démographique** : écart du taux de prédiction positive entre groupes (idéal : 0)
- **Ratio d'Impact Disproportionné** : rapport des taux entre groupe défavorisé et favorisé (règle des 4/5 : ≥ 0.8)
- **Égalité des Chances** : comparaison des True Positive Rates par groupe
- **Analyse par tranches d'âge** : parité sur 6 tranches (< 25 à 65+)

## Résultats Clés

| Biais | Métrique | Valeur | Statut |
|-------|----------|--------|--------|
| Genre | Parité Démographique | ~0.20 | Significatif |
| Genre | Ratio DI (F/M) | ~0.36 | Discriminatoire |
| Race | Parité Démographique | ~0.26 | Significatif |
| Âge | Écart max tranches | ~0.38 | Significatif |

---

*Projet académique — Parcours A : Détection de Biais*
