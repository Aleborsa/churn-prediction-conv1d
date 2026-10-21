"""
Passo 0 - Análise Exploratória de Dados (EDA)

Este script foi utilizado para a investigação inicial da base de dados bruta
'Online Retail II' antes das etapas de limpeza e engenharia de atributos (RFM).

Objetivos principais desta EDA no contexto do projeto:
1. Analisar a estrutura e a volumetria dos registros distribuídos nas múltiplas abas
   da planilha (anos 2009-2010 e 2010-2011).
2. Mapear valores ausentes (NaN), com destaque para a ausência da coluna 'Customer ID',
   o que inviabiliza o rastreio do comportamento de recompra desses registros.
3. Identificar anomalias transacionais, como quantidades negativas (que representam
   devoluções/cancelamentos) e preços unitários inconsistentes (zerados ou negativos).
4. Fundamentar empiricamente as regras de negócio aplicadas no script subsequente de
   higienização da base (limpar_precos.py).

Uso:
    python src/eda/eda.py
    python src/eda/eda.py caminho/para/online_retail_II.xlsx

Ajuste a variável ARQUIVO_EXCEL abaixo se preferir fixar o caminho local da base.
"""

import os
import sys
import io
import contextlib

import pandas as pd
from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
ARQUIVO_EXCEL = str(RAIZ_PROJETO / "dados" / "online_retail_II.xlsx")
ARQUIVO_RELATORIO = str(RAIZ_PROJETO / "dados" / "relatorio_eda.txt")
GERAR_GRAFICOS = False
PASTA_GRAFICOS = str(RAIZ_PROJETO / "dados" / "graficos_eda")

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _secao(titulo: str) -> str:
    linha = "=" * 80
    return f"\n{linha}\n{titulo}\n{linha}\n"


def _subsecao(titulo: str) -> str:
    return f"\n--- {titulo} ---\n"


def _describe_df(df: pd.DataFrame) -> str:
    """describe() robusto para DataFrames com colunas mistas."""
    try:
        return df.describe(include="all").to_string()
    except Exception:
        return df.astype(str).describe(include="all").to_string()


def _resumo_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela resumo por coluna: tipo, nulos, % nulos, únicos, exemplos."""
    resumo = pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nao_nulos": df.notna().sum(),
        "nulos": df.isna().sum(),
        "pct_nulos": (df.isna().mean() * 100).round(2),
        "unicos": df.nunique(dropna=True),
    })
    resumo["exemplo"] = [
        df[col].dropna().iloc[0] if df[col].notna().any() else None
        for col in df.columns
    ]
    return resumo


def _analisar_dataframe(df: pd.DataFrame, nome: str, buffer: io.StringIO) -> None:
    """Executa a EDA de um DataFrame e escreve no buffer."""
    buffer.write(_secao(f"ABA / CONJUNTO: {nome}"))

    buffer.write(_subsecao("Dimensões"))
    buffer.write(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}\n")

    buffer.write(_subsecao("Colunas"))
    buffer.write(", ".join(map(str, df.columns)) + "\n")

    buffer.write(_subsecao("Tipos de dados"))
    buffer.write(df.dtypes.to_string() + "\n")

    buffer.write(_subsecao("Primeiras 5 linhas"))
    buffer.write(df.head().to_string() + "\n")

    buffer.write(_subsecao("Últimas 5 linhas"))
    buffer.write(df.tail().to_string() + "\n")

    buffer.write(_subsecao("Resumo por coluna"))
    buffer.write(_resumo_colunas(df).to_string() + "\n")

    buffer.write(_subsecao("Estatísticas descritivas"))
    buffer.write(_describe_df(df) + "\n")

    buffer.write(_subsecao("Valores nulos"))
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if nulos.empty:
        buffer.write("Nenhum valor nulo encontrado.\n")
    else:
        buffer.write(nulos.to_string() + "\n")

    buffer.write(_subsecao("Linhas duplicadas"))
    buffer.write(f"Total de linhas duplicadas: {df.duplicated().sum()}\n")

    buffer.write(_subsecao("Cardinalidade (nº de valores únicos por coluna)"))
    buffer.write(df.nunique(dropna=True).sort_values(ascending=False).to_string() + "\n")

    # Contagens de colunas categóricas (poucos valores únicos)
    buffer.write(_subsecao("Contagens de colunas categóricas (até 20 valores únicos)"))
    categoricas = [
        c for c in df.columns
        if df[c].nunique(dropna=True) <= 20 and df[c].dtype != "float64"
    ]
    if not categoricas:
        buffer.write("Nenhuma coluna categórica identificada.\n")
    else:
        for col in categoricas:
            buffer.write(f"\n[{col}]\n")
            buffer.write(df[col].value_counts(dropna=False).to_string() + "\n")

    # Correlação numérica
    buffer.write(_subsecao("Matriz de correlação (colunas numéricas)"))
    numericas = df.select_dtypes(include="number")
    if numericas.shape[1] >= 2:
        buffer.write(numericas.corr(numeric_only=True).round(3).to_string() + "\n")
    else:
        buffer.write("Menos de 2 colunas numéricas; correlação não calculada.\n")


def _gerar_graficos(df: pd.DataFrame, nome: str) -> None:
    """Gera histogramas e heatmap de correlação, se as libs existirem."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[aviso] matplotlib indisponível, pulando gráficos: {e}")
        return

    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    nome_seguro = "".join(c if c.isalnum() else "_" for c in str(nome))

    numericas = df.select_dtypes(include="number")
    if numericas.shape[1] > 0:
        axes = numericas.hist(figsize=(12, 8), bins=30)
        plt.suptitle(f"Histogramas - {nome}")
        plt.tight_layout()
        caminho = os.path.join(PASTA_GRAFICOS, f"hist_{nome_seguro}.png")
        plt.savefig(caminho, dpi=100)
        plt.close()
        print(f"[ok] gráfico salvo: {caminho}")

    if numericas.shape[1] >= 2:
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(numericas.corr(numeric_only=True), annot=True,
                        fmt=".2f", cmap="coolwarm", center=0)
            plt.title(f"Correlação - {nome}")
            plt.tight_layout()
            caminho = os.path.join(PASTA_GRAFICOS, f"corr_{nome_seguro}.png")
            plt.savefig(caminho, dpi=100)
            plt.close()
            print(f"[ok] gráfico salvo: {caminho}")
        except Exception as e:
            print(f"[aviso] seaborn indisponível, pulando heatmap: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    caminho = sys.argv[1] if len(sys.argv) > 1 else ARQUIVO_EXCEL

    if not os.path.exists(caminho):
        print(f"[erro] arquivo não encontrado: {caminho}")
        print("Uso: python main.py caminho/para/arquivo.xlsx")
        sys.exit(1)

    print(f"Lendo arquivo: {caminho}")
    abas = pd.read_excel(caminho, sheet_name=None)  # dict {nome_aba: DataFrame}
    print(f"Abas encontradas: {list(abas.keys())}")

    buffer = io.StringIO()
    buffer.write(_secao("RELATÓRIO DE ANÁLISE EXPLORATÓRIA DE DADOS (EDA)"))
    buffer.write(f"Arquivo: {caminho}\n")
    buffer.write(f"Abas: {list(abas.keys())}\n")

    # EDA por aba
    for nome_aba, df in abas.items():
        _analisar_dataframe(df, nome_aba, buffer)
        if GERAR_GRAFICOS:
            _gerar_graficos(df, nome_aba)

    # EDA combinada (empilhando as abas)
    if len(abas) > 1:
        buffer.write(_secao("ANÁLISE COMBINADA (todas as abas empilhadas)"))
        partes = []
        for nome_aba, df in abas.items():
            tmp = df.copy()
            tmp["origem_aba"] = nome_aba
            partes.append(tmp)
        combinado = pd.concat(partes, ignore_index=True, sort=False)
        _analisar_dataframe(combinado, "COMBINADO", buffer)
        if GERAR_GRAFICOS:
            _gerar_graficos(combinado, "COMBINADO")

    relatorio = buffer.getvalue()

    # Imprime no console e salva em arquivo
    print(relatorio)
    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(f"\n[ok] relatório salvo em: {ARQUIVO_RELATORIO}")


if __name__ == "__main__":
    main()
