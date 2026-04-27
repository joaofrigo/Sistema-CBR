# PROBLEMA (descrição do caso clínico)
problema_numericos = [
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
]

problema_ordinais = [
    "social_support",
    "physical_activity",
    "work_or_study_impairment",
    "substance_use_risk",
    "clinical_severity"
]

problema_binarios = [
    "panic_symptoms",
    "concentration_difficulty",
]

problema_categoricos = [
    "gender",
    "appetite_change",
    "prior_treatment",
    "trauma_history",
    "current_medication"
]

problema_texto = [
    "main_issue",
    "comorbid_profile",
]

# SOLUÇÃO (intervenção/recomendação)
solucao_categoricos = [
    "intervention_type",
]

solucao_numericos = [
    "intensity",
    "weekly_frequency"
]

solucao_texto = [
    "recommendation_text"
]

solucao_ordinais = [
    
]
'''
NUMÉRICO      → linear
ORDINAL       → table
BINÁRIO       → equality
CATEGÓRICO    → equality
TEXTO LIVRE   → levenshtein ou embeddings

o esperado:

problema_numericos = [
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
    "work_or_study_impairment"
]

problema_ordinais = [
    "social_support",
    "physical_activity",
    "clinical_severity"
]

problema_binarios = [
    "panic_symptoms",
    "concentration_difficulty",
    "prior_treatment",
    "trauma_history"
]

problema_categoricos = [
    "gender",
    "appetite_change",
    "current_medication",
    "substance_use_risk",
    "comorbid_profile"
]

problema_texto = [
    "main_issue"
]

# SOLUÇÃO 

solucao_categoricos = [
    "intervention_type"
]

solucao_numericos = [
    "intensity",
    "weekly_frequency"
]

solucao_texto = [
    "recommendation_text"
]
'''