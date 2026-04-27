import numpy as np
from sklearn.tree import DecisionTreeClassifier


# =========================
# ENCODING (conforme config)
# =========================

MAP_BINARIO = {
    "yes": 1, "no": 0,
    "true": 1, "false": 0
}

MAP_SOCIAL_SUPPORT = {
    "low": 0,
    "moderate": 1,
    "high": 2
}

MAP_PHYSICAL_ACTIVITY = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3
}

MAP_WORK_IMPAIRMENT = {
    "low": 0,
    "moderate": 1,
    "high": 2
}

MAP_SUBSTANCE_RISK = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3
}

MAP_CLINICAL_SEVERITY = {
    "low": 0,
    "moderate": 1,
    "high": 2
}


def encode(case):
    return [
        # NUMÉRICOS
        case.age,
        case.anxiety_score,
        case.depression_score,
        case.stress_level,
        case.sleep_hours,
        case.symptom_duration_months,
        case.gad7_estimate,
        case.phq9_estimate,
        case.irritability_level,
        case.bmi_estimate,
        case.sleep_quality,

        # ORDINAIS
        MAP_SOCIAL_SUPPORT.get(case.social_support, 0),
        MAP_PHYSICAL_ACTIVITY.get(case.physical_activity, 0),
        MAP_WORK_IMPAIRMENT.get(case.work_or_study_impairment, 0),
        MAP_SUBSTANCE_RISK.get(case.substance_use_risk, 0),
        MAP_CLINICAL_SEVERITY.get(case.clinical_severity, 0),

        # BINÁRIOS
        MAP_BINARIO.get(case.panic_symptoms, 0),
        MAP_BINARIO.get(case.concentration_difficulty, 0),

        # CATEGÓRICOS (hash controlado)
        hash(case.gender) % 10,
        hash(case.appetite_change) % 10,
        hash(case.prior_treatment) % 10,
        hash(case.trauma_history) % 10,
        hash(case.current_medication) % 10,

        # TEXTO (sinal fraco estrutural)
        len(case.main_issue),
        len(case.comorbid_profile)
    ]


# =========================
# TREINO DA ÁRVORE
# =========================

def treinar_arvore_pesos(casebase):
    X = []
    y = []

    for case in casebase.values():
        X.append(encode(case))
        y.append(case.intervention_type)

    X = np.array(X)

    tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=42
    )

    tree.fit(X, y)

    return tree


# =========================
# EXTRAÇÃO DE PESOS
# =========================

def extrair_pesos(tree):
    features = [
        # numéricos
        "age",
        "anxiety_score",
        "depression_score",
        "stress_level",
        "sleep_hours",
        "symptom_duration_months",
        "gad7_estimate",
        "phq9_estimate",
        "irritability_level",
        "bmi_estimate",
        "sleep_quality",

        # ordinais
        "social_support",
        "physical_activity",
        "work_or_study_impairment",
        "substance_use_risk",
        "clinical_severity",

        # binários
        "panic_symptoms",
        "concentration_difficulty",

        # categóricos
        "gender",
        "appetite_change",
        "prior_treatment",
        "trauma_history",
        "current_medication",

        # texto
        "main_issue",
        "comorbid_profile"
    ]

    importances = tree.feature_importances_

    return {
        features[i]: float(importances[i])
        for i in range(len(features))
    }