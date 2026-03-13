import numpy as np
import pandas as pd


def demographic_parity_difference(y_true, y_pred, sensitive_attribute):
    """
    Calcule la différence de parité démographique entre groupes.
    Valeur idéale : 0 (pas de différence entre groupes)
    """
    groups = np.unique(sensitive_attribute)
    rates = {}
    for group in groups:
        mask = sensitive_attribute == group
        if mask.sum() > 0:
            rates[group] = y_pred[mask].mean()

    if len(rates) < 2:
        return {"difference": 0, "rates": rates, "groups": list(groups)}

    values = list(rates.values())
    difference = max(values) - min(values)

    return {
        "difference": difference,
        "rates": rates,
        "groups": list(groups),
        "max_group": max(rates, key=rates.get),
        "min_group": min(rates, key=rates.get),
    }


def disparate_impact_ratio(
    y_true, y_pred, sensitive_attribute, unprivileged_value, privileged_value
):
    """
    Calcule le ratio d'impact disproportionné.
    Valeur idéale : 1.0
    Règle des 4/5 : ratio < 0.8 indique une discrimination
    """
    mask_unpriv = sensitive_attribute == unprivileged_value
    mask_priv = sensitive_attribute == privileged_value

    rate_unpriv = y_pred[mask_unpriv].mean() if mask_unpriv.sum() > 0 else 0
    rate_priv = y_pred[mask_priv].mean() if mask_priv.sum() > 0 else 1

    ratio = rate_unpriv / rate_priv if rate_priv > 0 else 0

    return {
        "ratio": ratio,
        "rate_unprivileged": rate_unpriv,
        "rate_privileged": rate_priv,
        "unprivileged_value": unprivileged_value,
        "privileged_value": privileged_value,
        "is_biased": ratio < 0.8,
    }


def equal_opportunity_difference(y_true, y_pred, sensitive_attribute):
    """
    Calcule la différence d'égalité des chances (True Positive Rate par groupe).
    Valeur idéale : 0
    """
    groups = np.unique(sensitive_attribute)
    tpr = {}
    for group in groups:
        mask = sensitive_attribute == group
        positives = (y_true[mask] == 1)
        if positives.sum() > 0:
            tpr[group] = y_pred[mask][positives].mean()
        else:
            tpr[group] = 0

    if len(tpr) < 2:
        return {"difference": 0, "tpr": tpr}

    values = list(tpr.values())
    difference = max(values) - min(values)

    return {
        "difference": difference,
        "tpr": tpr,
        "max_group": max(tpr, key=tpr.get),
        "min_group": min(tpr, key=tpr.get),
    }


def compute_age_group_fairness(df, target_col, age_col, bins=None, labels=None):
    """
    Analyse la parité par tranches d'âge.
    """
    if bins is None:
        bins = [0, 25, 35, 45, 55, 65, 100]
    if labels is None:
        labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

    df = df.copy()
    df["age_group"] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)

    result = (
        df.groupby("age_group", observed=False)[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: "positive_rate"})
    )
    result["positive_rate"] = result["positive_rate"].round(4)
    return result
