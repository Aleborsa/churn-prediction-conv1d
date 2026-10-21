"""
Passo 2.1 - Construção da Variável-Alvo (Target)

Adiciona uma coluna binária à matriz RFM indicando se o cliente realizou
novas compras durante a janela temporal isolada no cálculo base (ex: últimos 90 dias).

Este script complementa o 'engenharia_rfm.py'. Ele cruza os clientes da base RFM
com as transações ocorridas no período de validação futuro, estabelecendo o
gabarito (ground truth) supervisionado para a rede neural:
    1 -> Cliente recomprou no período futuro (corte, data_max]
    0 -> Cliente não recomprou no período futuro (abandono/churn)

Uso:
    python src/cnn/adicionar_compra_futura.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from engenharia_rfm import (
    ARQUIVO_ENTRADA,
    ARQUIVO_SAIDA,
    NOME_ABA_SAIDA,
    COL_CLIENTE,
    COL_DATA,
    DIAS_IGNORADOS,
    carregar_vendas,
)

COL_COMPROU = f"Comprou_{DIAS_IGNORADOS}d"


def clientes_que_compraram_no_futuro(df: pd.DataFrame, dias_ignorados: int) -> set:
    """
    Retorna o conjunto de Customer IDs que fizeram ao menos uma compra
    no período (corte, data_max], onde corte = data_max - dias_ignorados.
    """
    data_max = df[COL_DATA].max()
    corte = data_max - pd.Timedelta(days=dias_ignorados)

    df_futuro = df[(df[COL_DATA] > corte) & (df[COL_DATA] <= data_max)]
    return set(df_futuro[COL_CLIENTE].unique())


def adicionar_flag_compra(rfm: pd.DataFrame, clientes: set) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm[COL_COMPROU] = rfm["CustomerID"].isin(clientes).astype(int)
    return rfm


def main() -> None:
    try:
        df = carregar_vendas(ARQUIVO_ENTRADA)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)

    caminho_rfm = Path(ARQUIVO_SAIDA)
    if not caminho_rfm.exists():
        print(
            f"[ERRO] Arquivo RFM não encontrado: {caminho_rfm}. "
            f"Execute engenharia_rfm.py primeiro.",
            file=sys.stderr,
        )
        sys.exit(1)

    rfm = pd.read_excel(caminho_rfm, sheet_name=NOME_ABA_SAIDA)

    clientes = clientes_que_compraram_no_futuro(df, DIAS_IGNORADOS)
    rfm = adicionar_flag_compra(rfm, clientes)

    with pd.ExcelWriter(caminho_rfm, engine="openpyxl") as writer:
        rfm.to_excel(writer, sheet_name=NOME_ABA_SAIDA, index=False)

    positivos = int(rfm[COL_COMPROU].sum())
    print(
        f"Coluna '{COL_COMPROU}' adicionada: "
        f"{positivos}/{len(rfm)} clientes compraram nos {DIAS_IGNORADOS} dias futuros."
    )
    print(f"Arquivo atualizado: {caminho_rfm} (aba '{NOME_ABA_SAIDA}')")


if __name__ == "__main__":
    main()
