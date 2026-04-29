import cbrkit
from typing import Any, Dict
from cbr_similarity import build_problem_similarity, build_solution_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class CBRCycle:
    def __init__(self, casebase: Any) -> None:
        """
        Inicializa o ciclo CBR, configurando as medidas de similaridade para 
        problemas e soluções, e constrói os motores de recuperação e reuso.
        """
        self.casebase = casebase

        # Espaço de problemas (retrieval)
        self.problem_similarity = build_problem_similarity(casebase)

        # Espaço de soluções (reuse)
        self.solution_similarity = build_solution_similarity()

        self.retriever = self._build_retriever()
        self.reuser = self._build_reuser()

    # =========================
    # RETRIEVAL 
    # =========================

    def retrieve(self, query: Dict[str, Any], casebase_externo: Any = None):
        """
        Interface principal para a fase de recuperação. Seleciona a base de casos 
        ativa e aplica o motor de busca para encontrar casos similares à query.
        """
        active_casebase = casebase_externo if casebase_externo is not None else self.casebase
        return self.retriever(active_casebase, query)

    def _build_retriever(self):
        """
        Constrói e configura internamente o motor de recuperação (retriever). 
        Inclui lógica para filtrar a própria query da base de casos durante testes.
        """
        def retriever(casebase, query):
            base = cbrkit.retrieval.build(self.problem_similarity)
            result = cbrkit.retrieval.apply_query(
                casebase=casebase,
                query=query,
                retrievers=base
            )

            step = result.final_step
            qr = step.default_query

            query_id = None
            for cid, case in qr.casebase.items():
                if case == query:
                    query_id = cid
                    break

            if query_id is None:
                return result

            new_ranking = [cid for cid in qr.ranking if cid != query_id]
            new_similarities = {cid: sim for cid, sim in qr.similarities.items() if cid != query_id}
            new_casebase = {cid: case for cid, case in qr.casebase.items() if cid != query_id}

            new_qr = qr.model_copy(update={
                "ranking": new_ranking,
                "similarities": new_similarities,
                "casebase": new_casebase
            })

            new_step = step.model_copy(update={"queries": {"default": new_qr}})
            return result.model_copy(update={"steps": [new_step]})

        return retriever

    # =========================
    # REUSE
    # =========================

    def _build_reuser(self):
        """
        Constrói o motor de reuso, definindo a função de adaptação clínica que ajusta 
        intensidade e frequência da solução com base na gravidade do novo caso.
        """
        def adaptation(case, query):
            mapa = {"low": 1, "moderate": 2, "high": 3}
            
            q = mapa.get(getattr(query, "clinical_severity", "moderate"), 2)
            c = mapa.get(getattr(case, "clinical_severity", "moderate"), 2)

            intensity = case.intensity
            weekly_frequency = case.weekly_frequency

            base_boost = (q - 1)
            case_penalty = (c - 1)

            intensity = intensity + base_boost - 0.5 * case_penalty
            weekly_frequency = weekly_frequency + base_boost - 0.5 * case_penalty

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
            return cbrkit.reuse.apply_result(result=result, reusers=reuser)

        return reuser_wrapper

    # =========================
    # MÉTODOS DE EXECUÇÃO 
    # =========================

    def run_retrieval(self, query: Any, casebase_externo: Any = None):
        """
        Executa isoladamente a fase de recuperação para uma dada consulta.
        """
        return self.retrieve(query, casebase_externo=casebase_externo)

    def run_reuse(self, retrieval_result: Any):
        """
        Executa isoladamente a fase de reuso/adaptação a partir de um resultado de recuperação.
        """
        return self.reuser(retrieval_result)

    def run_full(self, query: Any, casebase_externo: Any = None):
        """
        Orquestra o ciclo completo (Retrieval + Reuse) e extrai a solução adaptada 
        do caso melhor ranqueado.
        """
        retrieval_result = self.run_retrieval(query, casebase_externo=casebase_externo)
        reuse_result = self.run_reuse(retrieval_result)

        # 1. Acessamos o último passo do pipeline (Result -> ResultStep)
        step = reuse_result.final_step
        
        # 2. Pegamos o ID do melhor caso ranqueado (via default_query do step)
        best_id = step.ranking[0]
        return step.casebase[best_id]

    def run_loo(self) -> Dict[str, Any]:
        """
        Realiza a validação Leave-One-Out (LOO) em toda a base de casos, calculando 
        métricas de acurácia de classificação e erros médios de adaptação (MAE).
        """
        y_true_type = []
        y_pred_type = []
        errors_intensity = []
        errors_frequency = []
        
        # Lista para armazenar metadados de cada iteração para o teste consumir
        logs_detalhados = []
        
        items = list(self.casebase.items())
        print(f"Processando LOO para {len(items)} casos...")

        for case_id, query_case in items:
            # 1. Setup da base de treino (Exclui o caso atual)
            base_treino = {cid: c for cid, c in self.casebase.items() if cid != case_id}

            try:
                # 2. Execução Manual do Ciclo para capturar rastreabilidade
                # Retrieval
                retrieval_result = self.run_retrieval(query_case, casebase_externo=base_treino)
                step = retrieval_result.final_step
                
                # 1. Recupera o índice do caso com maior score de similaridade global (Nearest Neighbor)
                best_id = step.ranking[0]

                # 2. Extrai o grau de confiança (0 a 1) da correspondência encontrada para fins de teste
                similarity = step.similarities[best_id]

                # 3. Instancia o objeto completo do caso vencedor para servir de base para a fase de Adaptacão (Reuse)
                winner_case = base_treino[best_id]
                
                # Adaptação
                reuse_result = self.run_reuse(retrieval_result)
                solucao_prevista = reuse_result.final_step.casebase[best_id]

                # 3. Coleta de dados para métricas
                y_true_type.append(query_case.intervention_type)
                y_pred_type.append(solucao_prevista["intervention_type"])
                
                err_int = abs(query_case.intensity - solucao_prevista["intensity"])
                err_freq = abs(query_case.weekly_frequency - solucao_prevista["weekly_frequency"])
                
                errors_intensity.append(err_int)
                errors_frequency.append(err_freq)

                # 4. Registro de log para o teste consumir
                logs_detalhados.append({
                    "case_id": case_id,
                    "winner_id": best_id,
                    "similarity": float(similarity),
                    "expected_type": query_case.intervention_type,
                    "predicted_type": solucao_prevista["intervention_type"],
                    "query_main_issue": query_case.main_issue,
                    "winner_main_issue": winner_case.main_issue,
                    "is_correct": query_case.intervention_type == solucao_prevista["intervention_type"]
                })

            except Exception as e:
                print(f"Erro no processamento do caso {case_id}: {e}")

        # 5. Cálculo das Métricas Finais
        acc = accuracy_score(y_true_type, y_pred_type)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_type, y_pred_type, average='weighted', zero_division=0
        )

        stats = {
            "classification": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            },
            "adaptation_error": {
                "mae_intensity": np.mean(errors_intensity),
                "mae_frequency": np.mean(errors_frequency),
            },
            "total_cases": len(y_true_type),
            "logs": logs_detalhados 
        }

        return stats