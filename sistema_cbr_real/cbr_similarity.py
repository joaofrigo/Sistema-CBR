#cbr_similarity.py
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
    tree = treinar_arvore_pesos(casebase)
    raw = extrair_pesos(tree)

    alpha = 0.05

    return {
        k: (v + alpha) / (1 + alpha)
        for k, v in raw.items()
    }

def build_problem_similarity(casebase):
    weights = build_weights(casebase)
    return cbrkit.sim.attribute_value(
        attributes={
            **{a: cbrkit.sim.numbers.linear(max=100) for a in problema_numericos},

            **{a: cbrkit.sim.generic.table({
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
            }) for a in problema_ordinais},

            **{a: cbrkit.sim.generic.equality() for a in problema_binarios},

            **{a: cbrkit.sim.generic.equality() for a in problema_categoricos},

            **{a: cbrkit.sim.strings.levenshtein(case_sensitive=False) for a in problema_texto},
        },
        aggregator=cbrkit.sim.aggregator(
            "mean",
            pooling_weights=weights
        )
    )
    
def build_solution_similarity():
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