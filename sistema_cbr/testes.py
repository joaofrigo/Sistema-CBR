from main import BancoDeCasos
from cbr_similarity import build_problem_similarity, build_weights
from arvore_pesos import treinar_arvore_pesos, extrair_pesos
from cbr_cicle_kit import CBRCycle
import cbrkit
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def run_similarity_tests():
    db = BancoDeCasos("cbr_psychology_110_cases_clinical.csv")
    casebase = db.get_casebase()

    sim = build_problem_similarity(casebase)

    query = casebase[0]
    case = casebase[1]

    # 1. caso idêntico
    print("SELF SIMILARITY:")
    r = sim([(query, query)])
    print(r[0].value)

    # 2. comparação direta
    print("\nPAIR SIMILARITY:")
    r = sim([(query, case)])
    print(r[0].value)

    print("\nDETAILS:")
    print(r[0].attributes)

    # 3. variação na base
    print("\nRANK SAMPLE:")
    pairs = [(query, casebase[i]) for i in range(1, 6)]
    results = sim(pairs)

    for r in results:
        print(r.value)
        
def run_arvore_pesos_tests():
    db = BancoDeCasos("cbr_psychology_110_cases_clinical.csv")
    casebase = db.get_casebase()

    # 1. treino da árvore
    tree = treinar_arvore_pesos(casebase)

    # 2. extração de pesos
    weights = extrair_pesos(tree)

    print("\n=== PESOS APRENDIDOS ===\n")

    for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(k, "->", round(v, 6))

    # 3. verificação estrutural básica
    print("\n=== CHECK ===")

    total = sum(weights.values())
    print("Soma dos pesos:", total)

    print("Top 5 features:")
    top5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
    for k, v in top5:
        print(k, v)
      
        
def run_retrieval_debug():
    db = BancoDeCasos("cbr_psychology_110_cases_clinical.csv")
    casebase = db.get_casebase()

    from cbr_cicle_kit import CBRCycle

    cbr = CBRCycle(casebase)

    query = list(casebase.values())[5]

    result = cbr.retriever(casebase, query)

    print("\n=== RETRIEVAL DEBUG ===\n")

    ranking = list(result.ranking)
    scores = list(result.similarities)

    for i in range(min(5, len(ranking))):
        case_id = ranking[i]
        score = scores[i]

        print("rank:", i + 1)
        print("case_id:", case_id)
        print("score:", score)

        print("case:", casebase[case_id])

        print("------------------------")
        

def format_similarity(sim, weights, top_k=15):

    if sim is None:
        return None

    if not hasattr(sim, "attributes"):
        return float(sim)

    attrs = sim.attributes

    contrib = []

    for attr, score in attrs.items():
        w = weights.get(attr, 0.0)
        contrib.append((attr, float(score), float(w), float(score) * float(w)))

    contrib.sort(key=lambda x: x[3], reverse=True)

    top = contrib[:top_k]

    return {
        "global_similarity": float(sim.value),
        "top_contributors": [
            {
                "attribute": a,
                "similarity": round(s, 4),
                "weight": round(w, 4),
                "weighted_contribution": round(c, 4)
            }
            for a, s, w, c in top
        ]
    }

def debug_reuse(retrieval_result, query, weights, k=5):

    step = retrieval_result.final_step

    print("=== REUSE DEBUG ===")

    # =========================
    # RETRIEVAL
    # =========================
    ranking = step.ranking[:k]
    retrieved_cases = [step.casebase[r] for r in ranking]
    similarities = step.similarities

    print("\n--- GLOBAL STATS ---")
    print(f"n_retrieved = {len(retrieved_cases)}")

    if not retrieved_cases:
        print("EMPTY RETRIEVAL SET")
        return None

    # =========================
    # RETRIEVAL STATISTICS
    # =========================
    mean_intensity = sum(c.intensity for c in retrieved_cases) / len(retrieved_cases)
    mean_weekly = sum(c.weekly_frequency for c in retrieved_cases) / len(retrieved_cases)

    print("\n--- RETRIEVAL STATISTICS ---")
    print(f"mean_intensity_retrieved = {mean_intensity}")
    print(f"mean_weekly_retrieved = {mean_weekly}")

    mapa = {"low": 1, "moderate": 2, "high": 3}
    q_val = mapa.get(getattr(query, "clinical_severity", "moderate"), 2)

    results = []

    for i, case_id in enumerate(ranking):

        case = step.casebase[case_id]
        sim = similarities.get(case_id, None)

        print(f"\n==============================")
        print(f"CASE {i} | ID={case_id}")
        print(f"==============================")

        # =========================
        # TOP WEIGHTS
        # =========================
        print("\n--- TOP WEIGHTS ---")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for attr, w in sorted_weights[:10]:
            print(f"{attr:35s} weight={round(w, 4)}")

        # =========================
        # SIMILARITY
        # =========================
        sim_fmt = format_similarity(sim, weights)

        print("\n--- SIMILARITY ---")
        print(f"global_similarity = {sim_fmt['global_similarity']}")

        print("\n--- TOP CONTRIBUTORS (weighted impact) ---")
        for c in sim_fmt["top_contributors"]:
            print(
                f"{c['attribute']:35s} "
                f"sim={c['similarity']} "
                f"w={c['weight']} "
                f"impact={c['weighted_contribution']}"
            )

        # =========================
        # SEVERITY
        # =========================
        c_val = mapa.get(getattr(case, "clinical_severity", "moderate"), 2)

        print("\n--- SEVERITY COMPARISON ---")
        print(f"query_severity = {getattr(query, 'clinical_severity', 'moderate')}")
        print(f"case_severity  = {getattr(case, 'clinical_severity', 'moderate')}")
        print(f"query_numeric  = {q_val}")
        print(f"case_numeric   = {c_val}")
        print(f"delta          = {q_val - c_val}")

        # =========================
        # REUSE COMPONENTS
        # =========================
        alpha = 0.3

        retrieval_shift_intensity = alpha * (mean_intensity - case.intensity)
        retrieval_shift_weekly = alpha * (mean_weekly - case.weekly_frequency)

        severity_delta = q_val - c_val

        print("\n--- REUSE COMPONENTS ---")
        print(f"retrieval_shift_intensity = {retrieval_shift_intensity}")
        print(f"retrieval_shift_weekly    = {retrieval_shift_weekly}")
        print(f"severity_delta             = {severity_delta}")

        # =========================
        # APPLY REUSE
        # =========================
        intensity = case.intensity + retrieval_shift_intensity
        weekly_frequency = case.weekly_frequency + retrieval_shift_weekly

        intensity += severity_delta
        weekly_frequency += severity_delta

        intensity = int(round(max(1, min(5, intensity))))
        weekly_frequency = int(round(max(1, min(7, weekly_frequency))))

        print("\n--- FINAL RESULT ---")
        print(f"original_intensity = {case.intensity}")
        print(f"original_weekly    = {case.weekly_frequency}")
        print(f"adapted_intensity  = {intensity}")
        print(f"adapted_weekly     = {weekly_frequency}")

        results.append({
            "case_id": case_id,
            "similarity": sim,
            "original_intensity": case.intensity,
            "adapted_intensity": intensity,
            "original_weekly": case.weekly_frequency,
            "adapted_weekly": weekly_frequency,
        })

    return results

def teste_loo():
    # 1. Preparação
    db = BancoDeCasos("cbr_psychology_110_cases_clinical.csv")
    cycle = CBRCycle(casebase=db.casebase)
    
    y_true, y_pred = [], []
    errors_int, errors_freq = [], []
    scores = []
    
    # Lista para armazenar detalhes dos erros para análise posterior
    analise_falhas = []
    
    print(f"Iniciando teste de LOO com {len(db.casebase)} casos...")

    # 2. Loop Único de Auditoria
    for case_id, query_case in db.casebase.items():
        # Isola o caso (Base de treino exclui a query atual)
        base_treino = {cid: c for cid, c in db.casebase.items() if cid != case_id}
        
        # Auditoria de Recuperação (Retrieval)
        res_retrieval = cycle.run_retrieval(query_case, casebase_externo=base_treino)
        
        # Identificamos quem é o vizinho vencedor
        step = res_retrieval.final_step
        melhor_id = step.ranking[0]
        winner_case = base_treino[melhor_id]
        
        # Extração da similaridade (Lidando com objeto cbrkit ou float)
        sim_data = step.similarities[melhor_id]
        score = float(sim_data.value) if hasattr(sim_data, 'value') else float(sim_data)
        scores.append(score)

        # Auditoria de Adaptação (Reuse)
        res_reuse = cycle.run_reuse(res_retrieval)
        solucao_prevista = res_reuse.final_step.casebase[melhor_id]

        # Coleta de dados para estatísticas
        y_true.append(query_case.intervention_type)
        y_pred.append(solucao_prevista["intervention_type"])
        
        # Métricas de erro de intensidade/frequência
        errors_int.append(abs(query_case.intensity - solucao_prevista["intensity"]))
        errors_freq.append(abs(query_case.weekly_frequency - solucao_prevista["weekly_frequency"]))

        # 3. Registro de Auditoria (Se houver erro de classificação)
        if query_case.intervention_type != solucao_prevista["intervention_type"]:
            analise_falhas.append({
                "id_query": case_id,
                "id_winner": melhor_id,
                "sim": score,
                "txt_query": query_case.main_issue,
                "txt_winner": winner_case.main_issue,
                "esperado": query_case.intervention_type,
                "recebido": solucao_prevista["intervention_type"]
            })

    # 4. Cálculos Finais
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    distribuicao = dict(Counter(y_true))
    clones = sum(1 for s in scores if s >= 0.99)

    # 5. Relatório Consolidado
    print("\n" + "="*70)
    print("RELATÓRIO LOO")
    print("="*70)
    #print(f"Distribuição de Classes: {distribuicao}")
    print(f"Média de Similaridade: {np.mean(scores):.4f}")
    print(f"Casos com 'vizinhos idênticos' (Sim >= 0.99): {clones}")
    print("-" * 70)
    print(f"Acurácia:  {acc:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print(f"MAE Intensidade: {np.mean(errors_int):.2f}")
    print(f"MAE Frequência:  {np.mean(errors_freq):.2f}")
    print("-" * 70)

    # 6. Exibição dos Top 5 Erros Críticos (Onde a similaridade foi alta mas a classe errada)
    print("\nANÁLISE DE FALHAS (Top 5 Conflitos de Alta Similaridade):")
    # Ordena os erros por similaridade (do mais "confuso" para o menos)
    analise_falhas.sort(key=lambda x: x['sim'], reverse=True)
    
    for erro in analise_falhas[:5]:
        print(f"[{erro['id_query']} <-> {erro['id_winner']}] Sim: {erro['sim']:.4f}")
        print(f"  Q: '{erro['txt_query']}'")
        print(f"  W: '{erro['txt_winner']}'")
        print(f"  Classe: Esperava {erro['esperado']} mas veio {erro['recebido']}")
        print("-" * 40)
    
    print("="*70 + "\n")
    
    
#run_similarity_tests()
#run_arvore_pesos_tests()
#run_retrieval_debug()

query = type("Query", (), {
    "age": 29,
    "gender": "male",
    "anxiety_score": 7,
    "depression_score": 6,
    "sleep_quality": 7,
    "stress_level": 6,
    "social_support": "low",
    "physical_activity": "low",
    "main_issue": "persistent anxiety with work-related stress and sleep disturbance",
    "sleep_hours": 6.5,
    "symptom_duration_months": 8,
    "gad7_estimate": 11,
    "phq9_estimate": 13,
    "panic_symptoms": "no",
    "concentration_difficulty": "yes",
    "irritability_level": 6,
    "appetite_change": "stable",
    "prior_treatment": "none",
    "current_medication": "none",
    "trauma_history": "none",
    "substance_use_risk": "low",
    "work_or_study_impairment": "high",
    "bmi_estimate": 30.2,
    "comorbid_profile": "adjustment difficulties",
    "clinical_severity": "moderate"
})()

db = BancoDeCasos("cbr_psychology_110_cases_clinical.csv")
casebase = db.get_casebase()
cycle = CBRCycle(casebase)
retrieval_result = cycle.run_retrieval(query)
weights = build_weights(casebase)
#debug_reuse(retrieval_result, query, weights)

teste_loo()