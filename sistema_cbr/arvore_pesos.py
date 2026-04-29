import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# ENCODING CONFIGURATION
# =========================

MAP_BINARIO = {"yes": 1, "no": 0, "true": 1, "false": 0}

MAP_SOCIAL_SUPPORT = {"low": 0, "moderate": 1, "high": 2}
MAP_PHYSICAL_ACTIVITY = {"none": 0, "low": 1, "moderate": 2, "high": 3}
MAP_WORK_IMPAIRMENT = {"low": 0, "moderate": 1, "high": 2}
MAP_SUBSTANCE_RISK = {"none": 0, "low": 1, "moderate": 2, "high": 3}
MAP_CLINICAL_SEVERITY = {"low": 0, "moderate": 1, "high": 2}

# Encoders globais para consistência entre treino e extração
encoders = {}

def preparar_encoders(casebase):
    """
    Inicializa e treina os LabelEncoders para as variáveis categóricas do banco.
    Garante que strings sejam mapeadas para inteiros estáveis antes do treino da árvore.
    """
    campos_categoricos = [
        "gender", "appetite_change", "prior_treatment", 
        "trauma_history", "current_medication", "main_issue"
    ]
    
    for campo in campos_categoricos:
        le = LabelEncoder()
        # Coleta todos os valores únicos do banco para este campo
        valores = [getattr(c, campo) for c in casebase.values()]
        le.fit(valores)
        encoders[campo] = le

def encode_caso(case):
    """
    Realiza o pipeline de transformação de um objeto Caso para um vetor numérico.
    Aplica mapeamentos manuais para ordinais e LabelEncoding para categóricos.
    """
    return [
        # NUMÉRICOS (Pass-through)
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

        # ORDINAIS (Mapeamento Linear)
        MAP_SOCIAL_SUPPORT.get(case.social_support, 0),
        MAP_PHYSICAL_ACTIVITY.get(case.physical_activity, 0),
        MAP_WORK_IMPAIRMENT.get(case.work_or_study_impairment, 0),
        MAP_SUBSTANCE_RISK.get(case.substance_use_risk, 0),
        MAP_CLINICAL_SEVERITY.get(case.clinical_severity, 0),

        # BINÁRIOS
        MAP_BINARIO.get(case.panic_symptoms, 0),
        MAP_BINARIO.get(case.concentration_difficulty, 0),

        # CATEGÓRICOS (Label Encoding - Melhor que Hash)
        encoders["gender"].transform([case.gender])[0],
        encoders["appetite_change"].transform([case.appetite_change])[0],
        encoders["prior_treatment"].transform([case.prior_treatment])[0],
        encoders["trauma_history"].transform([case.trauma_history])[0],
        encoders["current_medication"].transform([case.current_medication])[0],
        
        # TEXTO CATEGÓRICO (Agora a árvore entende a CATEGORIA do problema)
        encoders["main_issue"].transform([case.main_issue])[0]
    ]

# =========================
# TREINO E EXTRAÇÃO
# =========================

def treinar_arvore_pesos(casebase):
    """
    Treina um classificador de Árvore de Decisão para aprender a importância 
    de cada atributo na predição do tipo de intervenção clínica.
    """
    # Prepara os encoders com base no estado atual do banco
    preparar_encoders(casebase)
    
    X = []
    y = []

    for case in casebase.values():
        X.append(encode_caso(case))
        y.append(case.intervention_type)

    X = np.array(X)

    # Hiperparâmetros da árvore
    tree = DecisionTreeClassifier(
            max_depth=5,
            max_features="sqrt",  # Força a árvore a tentar combinações diferentes de colunas
            min_samples_leaf=5,
            criterion="entropy",
            random_state=42
        )

    tree.fit(X, y)
    return tree

def extrair_pesos(tree):
    """
    Extrai o Gini Importance (ou Entropia) de cada feature da árvore treinada.
    Normaliza os valores para que a soma dos pesos seja exatamente 1.0.
    """
    features = [
        "age", "anxiety_score", "depression_score", "stress_level",
        "sleep_hours", "symptom_duration_months", "gad7_estimate",
        "phq9_estimate", "irritability_level", "bmi_estimate", "sleep_quality",
        "social_support", "physical_activity", "work_or_study_impairment",
        "substance_use_risk", "clinical_severity", "panic_symptoms",
        "concentration_difficulty", "gender", "appetite_change",
        "prior_treatment", "trauma_history", "current_medication", "main_issue"
    ]

    importances = tree.feature_importances_
    
    # Normalização garantida para soma = 1.0
    total = np.sum(importances)
    if total > 0:
        importances = importances / total

    return {
        features[i]: float(importances[i])
        for i in range(len(features))
    }