import cbrkit
from models import Case


class BancoDeCasos:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            casebase_raw = cbrkit.loaders.csv()(f)

        self.casebase = {
            k: Case.model_validate(self._normalize(v))
            for k, v in casebase_raw.items()
        }

    def _normalize(self, v: dict): # Existem algumas inconsistências como medium e moderate
        mapping = {
            "medium": "moderate",
            "mild": "moderate"
        }

        for k in v:
            if isinstance(v[k], str) and v[k] in mapping:
                v[k] = mapping[v[k]]

        return v

    def get_casebase(self):
        return self.casebase

    def get_case(self, idx: int):
        return self.casebase[idx]

    def build_query_from_case(self, idx: int):
        return self.casebase[idx]


db = BancoDeCasos("cbr_psychology_110_cases_clinical.csv")

casebase = db.get_casebase()
query = db.build_query_from_case(0)






'''
binary_sets = [{"yes", "no"}, {"true", "false"}, {"sim", "nao"}]
for a in similarity_config.problema_binarios:
    values = {str(getattr(case, a)).lower() for case in casebase.values()}

    is_binary = any(values <= s for s in binary_sets)

    print(a, values, "BINÁRIO" if is_binary else "NÃO BINÁRIO")

for a in similarity_config.problema_ordinais:
    values = {getattr(case, a) for case in casebase.values()}

    if values <= {"low", "medium", "high"}:
        print(a, "ORDINAL (compatível)")
    elif all(isinstance(v, (int, float)) for v in values):
        print(a, "NUMERIC (não ordinal)")
    else:
        print(a, "MIXED / inválido", values)
        
case = casebase[0]
for k, v in case.model_dump().items():
    if k in similarity_config.problema_numericos:
        print(k, type(v), v)
#print(casebase[0], "\n")
#print(query)

'''