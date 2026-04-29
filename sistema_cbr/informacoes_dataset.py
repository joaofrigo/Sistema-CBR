import pandas as pd
from similarity_config import (
    problema_numericos, 
    problema_ordinais, 
    problema_binarios, 
    problema_categoricos,
    solucao_numericos
)

def gerar_analise_descritiva(caminho_csv: str):
    """
    Gera uma análise completa baseada na estrutura definida no similarity_config.
    """
    df = pd.read_csv(caminho_csv)
    
    print("=== 1. ESTATÍSTICAS NUMÉRICAS (PROBLEMA) ===")
    # Usa a lista exata do seu config
    print(df[problema_numericos].describe().T)
    print("\n")

    print("=== 2. DISTRIBUIÇÃO DAS CLASSES (SOLUÇÃO) ===")
    counts = df["intervention_type"].value_counts()
    percs = df["intervention_type"].value_counts(normalize=True) * 100
    dist_intervencao = pd.DataFrame({'Total': counts, 'Porcentagem (%)': percs})
    print(dist_intervencao)
    print("\n")

    print("=== 3. ANÁLISE DE VARIÁVEIS CATEGÓRICAS E BINÁRIAS ===")
    # Identifica o valor mais comum (Moda) para entender o perfil padrão do banco
    cat_bin = problema_categoricos + problema_binarios
    for col in cat_bin:
        top_val = df[col].mode()[0]
        count_val = df[col].value_counts().iloc[0]
        print(f"{col.upper()}: Valor dominante é '{top_val}' ({count_val} casos)")
    print("\n")

    print("=== 4. PERFIL DE GRAVIDADE E VARIÁVEIS ORDINAIS ===")
    for col in problema_ordinais:
        print(f"Distribuição de {col}:")
        print(df[col].value_counts())
        print("-" * 20)
    print("\n")


if __name__ == "__main__":
    try:
        gerar_analise_descritiva("cbr_psychology_110_cases_clinical.csv")
    except FileNotFoundError:
        print("Erro: Arquivo CSV não encontrado.")