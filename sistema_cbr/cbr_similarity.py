# cbr_similarity.py
import cbrkit
from arvore_pesos import treinar_arvore_pesos, extrair_pesos
from similarity_config import (
    problema_numericos,
    problema_ordinais,
    problema_binarios,
    problema_categoricos,
    problema_texto
)

def build_weights(casebase):
    """
    Orquestra o treinamento da árvore de decisão e extrai a importância dos atributos.
    Aplica suavização (alpha) para garantir que atributos com importância zero na árvore
    ainda contribuam minimamente para o cálculo de similaridade.
    """
    tree = treinar_arvore_pesos(casebase)
    raw = extrair_pesos(tree)

    # Suavização para garantir que atributos ignorados pela árvore ainda existam
    alpha = 0.05 

    # a soma de todos os pesos deve ser exatamente 1.0
    suavizados = {k: v + alpha for k, v in raw.items()}
    total = sum(suavizados.values())
    
    return {k: v / total for k, v in suavizados.items()}

def build_problem_similarity(casebase):
    """
    Constrói a configuração de similaridade para o Espaço de Problemas (fase de Retrieval).
    Combina pesos dinâmicos da árvore com um peso fixo forçado para o campo 'main_issue',
    definindo as métricas específicas para dados numéricos, ordinais, categóricos e textuais.
    """
    
    # Obtém os pesos dinâmicos da árvore
    weights = build_weights(casebase)
    

    # O peso do main_issue é importante, mesmo que a árvore ache que não, logo:
    target_weight = 0.18
    if "main_issue" in weights:
        # Removemos o peso original para recalcular a proporção dos outros
        old_val = weights.pop("main_issue")
        total_others = sum(weights.values())
        
        # Re-normalizamos os outros pesos para que somem (1.0 - target_weight)
        # Mantendo a proporção de importância que a árvore deu a eles
        weights = {k: (v / total_others) * (1 - target_weight) for k, v in weights.items()}
        
        # Injetamos o peso dominante
        weights["main_issue"] = target_weight

    # Tabela de similaridade para campos ordinais (clínicos)
    tabela_ordinal = cbrkit.sim.generic.table({
        ("none", "none"): 1.0,
        ("none", "low"): 0.75,
        ("none", "moderate"): 0.25,
        ("none", "high"): 0.0,
        ("low", "none"): 0.75,
        ("low", "low"): 1.0,
        ("low", "moderate"): 0.5,
        ("low", "high"): 0.0,
        ("moderate", "none"): 0.25,
        ("moderate", "low"): 0.5,
        ("moderate", "moderate"): 1.0,
        ("moderate", "high"): 0.5,
        ("high", "none"): 0.0,
        ("high", "low"): 0.0,
        ("high", "moderate"): 0.5,
        ("high", "high"): 1.0,
    })

    return cbrkit.sim.attribute_value(
        attributes={
            # NUMÉRICOS: Distância linear (ajustado para max=100 conforme o seu padrão)
            **{a: cbrkit.sim.numbers.linear(max=100) for a in problema_numericos},

            # ORDINAIS: Mapeamento via tabela de proximidade
            **{a: tabela_ordinal for a in problema_ordinais},

            # BINÁRIOS/CATEGÓRICOS: Igualdade Exata (String ou Int)
            **{a: cbrkit.sim.generic.equality() for a in problema_binarios},
            **{a: cbrkit.sim.generic.equality() for a in problema_categoricos},

            # TEXTO: Levenshtein (Para main_issue e comorbid_profile)
            **{a: cbrkit.sim.strings.levenshtein(case_sensitive=False) for a in problema_texto},
        },
        aggregator=cbrkit.sim.aggregator(
            pooling="mean",
            pooling_weights=weights
        )
    )
    
    
def build_solution_similarity():
    """
    Define a medida de similaridade para o Espaço de Soluções.
    Utilizada principalmente para avaliar o sucesso de adaptações ou comparar 
    recomendações sugeridas com as reais do banco de casos.
    """
    weights = {
        "intervention_type": 1.0,
        "intensity": 0.8,
        "weekly_frequency": 0.8,
        "recommendation_text": 0.3
    }

    return cbrkit.sim.attribute_value(
        attributes={
            "intervention_type": cbrkit.sim.generic.equality(),

            "intensity": cbrkit.sim.numbers.linear(max=10),

            "weekly_frequency": cbrkit.sim.numbers.linear(max=7),

            "recommendation_text": cbrkit.sim.strings.levenshtein(case_sensitive=False)
        },
        aggregator=cbrkit.sim.aggregator(
            "mean",
            pooling_weights=weights
        )
    )