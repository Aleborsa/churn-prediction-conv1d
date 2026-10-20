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
    - Os últimos DIAS_IGNORADOS dias (contados a partir da data mais recente
      da base) são DESCONSIDERADOS no cálculo do RFM
    - Data de referência = data máxima (após o corte) do dataset
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

# Quantos dias (a partir da data mais recente da base) devem ser ignorados.
DIAS_IGNORADOS = 90

# Se None, usa a data máxima do dataset (após o corte) como referência.
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
# Corte temporal (ignorar os últimos N dias)
# ---------------------------------------------------------------------------
def _data_corte(df: pd.DataFrame, dias_ignorados: int) -> pd.Timestamp:
    """Retorna a data limite: data máxima da base menos `dias_ignorados`."""
    return df[COL_DATA].max() - pd.Timedelta(days=dias_ignorados)


def aplicar_corte_temporal(df: pd.DataFrame, dias_ignorados: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Remove os registros dos últimos `dias_ignorados` dias e retorna
    (dataframe_filtrado, data_de_corte).
    """
    corte = _data_corte(df, dias_ignorados)
    df_filtrado = df[df[COL_DATA] <= corte].copy()
    return df_filtrado, corte


# ---------------------------------------------------------------------------
# Cálculo do RFM
# ---------------------------------------------------------------------------
def calcular_rfm(df: pd.DataFrame, data_referencia: str | pd.Timestamp | None = None) -> pd.DataFrame:
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

    data_max = df[COL_DATA].max()

    # Ignora os últimos DIAS_IGNORADOS dias da base
    df, data_corte = aplicar_corte_temporal(df, DIAS_IGNORADOS)

    if df.empty:
        print(
            f"[ERRO] Após ignorar os últimos {DIAS_IGNORADOS} dias "
            f"(corte em {data_corte.date()}), não restaram registros.",
            file=sys.stderr,
        )
        sys.exit(1)

    # A referência de recência é a própria data de corte
    rfm = calcular_rfm(df, data_referencia=DATA_REFERENCIA or data_corte)
    salvar_rfm(rfm, ARQUIVO_SAIDA, NOME_ABA_SAIDA)

    print(f"Data máxima da base: {data_max.date()}")
    print(f"Corte (ignorando {DIAS_IGNORADOS} dias): {data_corte.date()}")
    print(f"RFM gerado com {len(rfm)} clientes -> {ARQUIVO_SAIDA} (aba '{NOME_ABA_SAIDA}')")


if __name__ == "__main__":
    main()
