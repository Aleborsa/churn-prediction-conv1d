"""
Engenharia de atributos — RFM
-----------------------------
Transforma uma base de vendas (uma linha por item de fatura) em uma tabela
com uma linha por cliente, contendo:

    R — Recency   : dias desde a última compra (menor = melhor)
    F — Frequency : nº de faturas distintas (Invoice) do cliente
    M — Monetary  : valor total gasto pelo cliente no período

Premissas (ajustáveis nas constantes abaixo):
    - Entrada em Excel (.xlsx) já tratada: online_retail_II_limpo.xlsx
    - TODAS as abas do arquivo são lidas e concatenadas antes do cálculo
    - Colunas: Customer ID, Invoice, InvoiceDate, Quantity, Price
    - O valor de cada linha é calculado como Quantity * Price
    - Data de referência = data máxima de InvoiceDate do dataset
    - Saída em Excel, uma aba "RFM"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
ARQUIVO_ENTRADA = "online_retail_II_limpo.xlsx"
ARQUIVO_SAIDA = "rfm.xlsx"
NOME_ABA_SAIDA = "RFM"

COL_CLIENTE = "Customer ID"
COL_FATURA = "Invoice"
COL_DATA = "InvoiceDate"
COL_QUANTIDADE = "Quantity"
COL_PRECO = "Price"

# Nome da coluna calculada internamente (Quantity * Price)
COL_VALOR = "ValorTotal"

# Se None, usa a data máxima do dataset como referência.
# Caso contrário, informe uma string "YYYY-MM-DD".
DATA_REFERENCIA: str | None = None


# ---------------------------------------------------------------------------
# Validações
# ---------------------------------------------------------------------------
def _validar_colunas(df: pd.DataFrame, colunas: list[str], nome_aba: str = "") -> None:
    faltantes = [c for c in colunas if c not in df.columns]
    if faltantes:
        origem = f" (aba '{nome_aba}')" if nome_aba else ""
        raise ValueError(
            f"Colunas ausentes no arquivo de entrada{origem}: {faltantes}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )


def _validar_coluna_numerica(df: pd.DataFrame, nome_coluna: str, nome_aba: str = "") -> None:
    if not pd.api.types.is_numeric_dtype(df[nome_coluna]):
        origem = f" (aba '{nome_aba}')" if nome_aba else ""
        raise TypeError(
            f"A coluna '{nome_coluna}'{origem} deve ser numérica, "
            f"mas veio como {df[nome_coluna].dtype}."
        )


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def carregar_vendas(caminho: str | Path) -> pd.DataFrame:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    # Lê TODAS as abas do arquivo como um dict {nome_aba: DataFrame}
    abas = pd.read_excel(caminho, sheet_name=None)

    if not abas:
        raise ValueError("O arquivo de entrada não possui abas para leitura.")

    colunas_obrigatorias = [COL_CLIENTE, COL_FATURA, COL_DATA, COL_QUANTIDADE, COL_PRECO]

    frames = []
    for nome_aba, df_aba in abas.items():
        _validar_colunas(df_aba, colunas_obrigatorias, nome_aba)
        _validar_coluna_numerica(df_aba, COL_QUANTIDADE, nome_aba)
        _validar_coluna_numerica(df_aba, COL_PRECO, nome_aba)
        df_aba = df_aba.copy()
        df_aba["origem_aba"] = nome_aba
        frames.append(df_aba)

    df = pd.concat(frames, ignore_index=True)

    # Normalizações básicas
    df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors="coerce")
    df = df.dropna(subset=[COL_CLIENTE, COL_DATA])

    # Remove devoluções / faturas canceladas (prefixo 'C')
    df = df[~df[COL_FATURA].astype(str).str.startswith("C")]

    # Cria o valor total de cada linha
    df[COL_VALOR] = df[COL_QUANTIDADE] * df[COL_PRECO]

    # Remove valores não positivos
    df = df[df[COL_VALOR] > 0]

    return df


# ---------------------------------------------------------------------------
# Cálculo do RFM
# ---------------------------------------------------------------------------
def calcular_rfm(df: pd.DataFrame, data_referencia: str | None = None) -> pd.DataFrame:
    if data_referencia is not None:
        ref = pd.Timestamp(data_referencia)
    else:
        ref = df[COL_DATA].max()

    agrupado = df.groupby(COL_CLIENTE).agg(
        ultima_compra=(COL_DATA, "max"),
        frequency=(COL_FATURA, "nunique"),
        monetary=(COL_VALOR, "sum"),
    )

    agrupado["recency"] = (ref - agrupado["ultima_compra"]).dt.days

    rfm = (
        agrupado[["recency", "frequency", "monetary"]]
        .rename(
            columns={
                "recency": "Recency",
                "frequency": "Frequency",
                "monetary": "Monetary",
            }
        )
        .reset_index()
        .rename(columns={COL_CLIENTE: "CustomerID"})
        .sort_values("Monetary", ascending=False)
        .reset_index(drop=True)
    )

    return rfm


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------
def salvar_rfm(rfm: pd.DataFrame, caminho: str | Path, nome_aba: str) -> None:
    caminho = Path(caminho)
    with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
        rfm.to_excel(writer, sheet_name=nome_aba, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        df = carregar_vendas(ARQUIVO_ENTRADA)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)

    rfm = calcular_rfm(df, DATA_REFERENCIA)
    salvar_rfm(rfm, ARQUIVO_SAIDA, NOME_ABA_SAIDA)

    print(f"RFM gerado com {len(rfm)} clientes -> {ARQUIVO_SAIDA} (aba '{NOME_ABA_SAIDA}')")


if __name__ == "__main__":
    main()
