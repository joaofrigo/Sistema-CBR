# cbr_cycle.py

import cbrkit
from cbr_similarity import build_problem_similarity, build_solution_similarity


class CBRCycle:

    def __init__(self, casebase):
        self.casebase = casebase

        # problem space (retrieval)
        self.problem_similarity = build_problem_similarity(casebase)

        # solution space (reuse)
        self.solution_similarity = build_solution_similarity()

        self.retriever = self._build_retriever()
        self.reuser = self._build_reuser()

    # =========================
    # RETRIEVAL
    # =========================
    def _build_retriever(self):

        def retriever(casebase, query):
            base = cbrkit.retrieval.build(self.problem_similarity)

            result = cbrkit.retrieval.apply_query(
                casebase=casebase,
                query=query,
                retrievers=base
            )

            step = result.final_step
            qr = step.default_query

            # identificar self-match e remover
            query_id = None
            for cid, case in qr.casebase.items():
                if case == query:
                    query_id = cid
                    break

            if query_id is None:
                return result

            new_ranking = [cid for cid in qr.ranking if cid != query_id]
            new_similarities = {
                cid: sim for cid, sim in qr.similarities.items()
                if cid != query_id
            }
            new_casebase = {
                cid: case for cid, case in qr.casebase.items()
                if cid != query_id
            }

            # criar novo QueryResultStep
            new_qr = qr.model_copy(
                update={
                    "ranking": new_ranking,
                    "similarities": new_similarities,
                    "casebase": new_casebase
                }
            )

            # criar novo ResultStep
            new_step = step.model_copy(
                update={
                    "queries": {"default": new_qr}
                }
            )

            # criar novo Result
            new_result = result.model_copy(
                update={
                    "steps": [new_step]
                }
            )

            return new_result

        return retriever

    # =========================
    # REUSE
    # =========================
    
    def _build_reuser(self):

        def adaptation(case, query):

            mapa = {"low": 1, "moderate": 2, "high": 3}

            q = mapa.get(getattr(query, "clinical_severity", "moderate"), 2)
            c = mapa.get(getattr(case, "clinical_severity", "moderate"), 2)

            intensity = case.intensity
            weekly_frequency = case.weekly_frequency

            # ajuste monotônico baseado em severidade da query
            # (query mais grave -> aumenta baseline)
            base_boost = (q - 1)

            # penalização leve por severidade do caso
            case_penalty = (c - 1)

            # combinação estável
            intensity = intensity + base_boost - 0.5 * case_penalty
            weekly_frequency = weekly_frequency + base_boost - 0.5 * case_penalty

            # clamp inteiro antes da saída
            intensity = int(round(max(1, min(5, intensity))))
            weekly_frequency = int(round(max(1, min(7, weekly_frequency))))

            return {
                "intervention_type": case.intervention_type,
                "intensity": intensity,
                "weekly_frequency": weekly_frequency,
                "recommendation_text": case.recommendation_text
            }

        reuser = cbrkit.reuse.build(
            adaptation_func=adaptation,
            similarity_func=self.solution_similarity
        )

        def reuser_wrapper(result):
            return cbrkit.reuse.apply_result(
                result=result,
                reusers=reuser
            )

        return reuser_wrapper

    # =========================
    # EXECUÇÃO
    # =========================
    def run_retrieval(self, query):
        return self.retriever(self.casebase, query)

    def run_reuse(self, retrieval_result):
        return self.reuser(retrieval_result)
    
    # =========================
    # PIPELINE COMPLETO
    # =========================
    def run_full(self, query):

        retrieval_result = self.run_retrieval(query)

        reuse_result = self.run_reuse(retrieval_result)

        qr = reuse_result.final_step.default_query

        best_id = qr.ranking[0]

        return qr.casebase[best_id]