"""
Passo 1 - Limpeza e Higienização de Dados

Remove linhas com anomalias de faturamento (Price <= 0.01), devoluções ou
cancelamentos (Quantity <= 0), registros não identificados (Customer ID vazio)
ou itens operacionais sem valor comercial (StockCode em lista de exclusão).

O processamento ocorre em um arquivo Excel com múltiplas abas (anos diferentes),
gerando um novo arquivo particionado: abas "limpas_" para uso no cálculo do RFM
e abas "removidas_" para auditoria e prestação de contas dos dados filtrados.

Uso:
    python src/limpeza_dados/limpar_precos.py
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import xlsxwriter

# ---------------------------------------------------------------------------
# Configuração de Caminhos
# ---------------------------------------------------------------------------
RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
PASTA_DADOS = RAIZ_PROJETO / "dados"

ARQUIVO_ENTRADA = str(PASTA_DADOS / "online_retail_II.xlsx")
ARQUIVO_SAIDA = str(PASTA_DADOS / "online_retail_II_limpo.xlsx")

# ---------------------------------------------------------------------------
# Parâmetros de Limpeza
# ---------------------------------------------------------------------------
COLUNA_PRECO = "Price"              # nome da coluna de preço
LIMITE_PRECO = 0.01                 # remove linhas com preço <= este valor
COLUNA_QUANTIDADE = "Quantity"      # nome da coluna de quantidade
LIMITE_QUANTIDADE = 0               # remove linhas com quantidade <= este valor
COLUNA_CUSTOMER_ID = "Customer ID"  # nome da coluna de cliente
COLUNA_STOCK_CODE = "StockCode"     # nome da coluna de código de estoque
PREFIXO_REMOVIDAS = "removidas"
PREFIXO_LIMPAS = "limpas"

# StockCodes considerados inválidos (linhas com esses códigos são removidas).
# Lista deduplicada a partir dos valores informados.
STOCK_CODES_INVALIDOS = {
    "POST",
    "PADS",
    "M",
    "gift_0001_90",
    "gift_0001_80",
    "gift_0001_70",
    "gift_0001_60",
    "gift_0001_50",
    "gift_0001_40",
    "gift_0001_30",
    "gift_0001_20",
    "gift_0001_10",
    "GIFT",
    "DOT",
    "D",
    "CRUK",
    "C3",
    "C2",
    "BANK CHARGES",
    "B",
    "AMAZONFEE",
    "ADJUST2",
    "ADJUST",
    "TEST002",
    "TEST001",
    "S",
}

# Limite do Excel por aba
MAX_LINHAS_EXCEL = 1_048_576

# Progresso a cada N linhas (0 = desativa)
PROGRESSO_A_CADA = 100_000


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------
def _validar_coluna_numerica(df: pd.DataFrame, nome_coluna: str, nome_aba: str) -> None:
    """Garante que a coluna existe e é numérica."""
    if nome_coluna not in df.columns:
        raise KeyError(
            f"A coluna '{nome_coluna}' não existe na aba '{nome_aba}'. "
            f"Colunas disponíveis: {list(df.columns)}"
        )
    if not pd.api.types.is_numeric_dtype(df[nome_coluna]):
        raise TypeError(
            f"A coluna '{nome_coluna}' da aba '{nome_aba}' não é numérica "
            f"(tipo atual: {df[nome_coluna].dtype})."
        )


def _validar_coluna_existente(df: pd.DataFrame, nome_coluna: str, nome_aba: str) -> None:
    """Garante que a coluna existe (sem exigir tipo específico)."""
    if nome_coluna not in df.columns:
        raise KeyError(
            f"A coluna '{nome_coluna}' não existe na aba '{nome_aba}'. "
            f"Colunas disponíveis: {list(df.columns)}"
        )


def _sanitizar_nome_aba(nome: str) -> str:
    """
    Ajusta o nome para ser aceito pelo Excel:
    - remove caracteres proibidos: : \\ / ? * [ ]
    - limita a 31 caracteres
    """
    nome = re.sub(r"[:\\/?*\[\]]", "_", str(nome))
    return nome[:31]


def _nome_aba_saida(prefixo: str, nome_origem: str) -> str:
    """Monta o nome da aba de saída respeitando o limite de 31 caracteres."""
    espaco = 31 - len(prefixo) - 1
    base = _sanitizar_nome_aba(nome_origem)[:espaco]
    return f"{prefixo}_{base}"


def separar_por_aba(abas: dict) -> tuple:
    """
    Recebe um dict {nome_aba: DataFrame} e retorna uma tupla
    (saidas, estatisticas):

        saidas: dict {nome_aba_saida: DataFrame} com as abas de saída
                já separadas por ano. Para cada aba de origem são geradas
                2 entradas:
                    - "limpas_<aba>"
                    - "removidas_<aba>"

        estatisticas: dict com os totais consolidados de todas as abas:
                    - total_linhas
                    - total_removidas
                    - total_por_preco
                    - total_por_quantidade
                    - total_por_customer_vazio
                    - total_por_stock_code
                    - total_por_ambos
                    - total_limpas

    Critérios de remoção (vai para "removidas" se atender QUALQUER um):
        - Price <= LIMITE_PRECO
        - Quantity <= LIMITE_QUANTIDADE
        - Customer ID vazio (NaN/None ou string vazia)
        - StockCode na lista STOCK_CODES_INVALIDOS
    """
    saidas = {}

    total_linhas = 0
    total_removidas = 0
    total_por_preco = 0
    total_por_quantidade = 0
    total_por_customer_vazio = 0
    total_por_stock_code = 0
    total_por_ambos = 0
    total_limpas = 0

    for nome_aba, df in abas.items():
        _validar_coluna_numerica(df, COLUNA_PRECO, nome_aba)
        _validar_coluna_numerica(df, COLUNA_QUANTIDADE, nome_aba)
        _validar_coluna_existente(df, COLUNA_CUSTOMER_ID, nome_aba)
        _validar_coluna_existente(df, COLUNA_STOCK_CODE, nome_aba)

        # Máscaras individuais (para log)
        mascara_preco = df[COLUNA_PRECO] <= LIMITE_PRECO
        mascara_qtd = df[COLUNA_QUANTIDADE] <= LIMITE_QUANTIDADE

        # Customer ID vazio: NaN/None ou string vazia (após strip)
        col_customer = df[COLUNA_CUSTOMER_ID]
        mascara_customer_vazio = col_customer.isna()
        if col_customer.dtype == object:
            mascara_customer_vazio = mascara_customer_vazio | (
                col_customer.astype(str).str.strip() == ""
            )

        # StockCode inválido: valor (após strip) presente na lista
        col_stock = df[COLUNA_STOCK_CODE].fillna("")
        mascara_stock_code = col_stock.astype(str).str.strip().str.upper().isin(
            {c.upper() for c in STOCK_CODES_INVALIDOS}
        )

        # Máscara combinada: inválida se atender QUALQUER critério
        mascara_invalida = (
            mascara_preco
            | mascara_qtd
            | mascara_customer_vazio
            | mascara_stock_code
        )

        removidas = df[mascara_invalida].copy()
        limpas = df[~mascara_invalida].copy()

        removidas["origem_aba"] = nome_aba
        limpas["origem_aba"] = nome_aba

        nome_limpas = _nome_aba_saida(PREFIXO_LIMPAS, nome_aba)
        nome_removidas = _nome_aba_saida(PREFIXO_REMOVIDAS, nome_aba)

        if len(limpas) > MAX_LINHAS_EXCEL - 1:
            raise ValueError(
                f"A aba '{nome_limpas}' teria {len(limpas)} linhas, "
                f"acima do limite do Excel ({MAX_LINHAS_EXCEL - 1})."
            )
        if len(removidas) > MAX_LINHAS_EXCEL - 1:
            raise ValueError(
                f"A aba '{nome_removidas}' teria {len(removidas)} linhas, "
                f"acima do limite do Excel ({MAX_LINHAS_EXCEL - 1})."
            )

        saidas[nome_limpas] = limpas
        saidas[nome_removidas] = removidas

        # Acumula estatísticas globais
        qtd_preco = int(mascara_preco.sum())
        qtd_qtd = int(mascara_qtd.sum())
        qtd_customer_vazio = int(mascara_customer_vazio.sum())
        qtd_stock_code = int(mascara_stock_code.sum())
        qtd_ambos = int(
            (
                mascara_preco
                & mascara_qtd
                & mascara_customer_vazio
                & mascara_stock_code
            ).sum()
        )

        total_linhas += len(df)
        total_removidas += len(removidas)
        total_por_preco += qtd_preco
        total_por_quantidade += qtd_qtd
        total_por_customer_vazio += qtd_customer_vazio
        total_por_stock_code += qtd_stock_code
        total_por_ambos += qtd_ambos
        total_limpas += len(limpas)

        print(
            f"[aba: {nome_aba}] total={len(df)} | "
            f"removidas={len(removidas)} -> '{nome_removidas}' | "
            f"limpas={len(limpas)} -> '{nome_limpas}'"
        )
        print(
            f"    detalhe remoção: "
            f"preço<={LIMITE_PRECO}: {qtd_preco} | "
            f"qtd<={LIMITE_QUANTIDADE}: {qtd_qtd} | "
            f"customer vazio: {qtd_customer_vazio} | "
            f"stock code inválido: {qtd_stock_code} | "
            f"todos os 4: {qtd_ambos}"
        )

    estatisticas = {
        "total_linhas": total_linhas,
        "total_removidas": total_removidas,
        "total_por_preco": total_por_preco,
        "total_por_quantidade": total_por_quantidade,
        "total_por_customer_vazio": total_por_customer_vazio,
        "total_por_stock_code": total_por_stock_code,
        "total_por_ambos": total_por_ambos,
        "total_limpas": total_limpas,
    }

    return saidas, estatisticas


def _escrever_dataframe(worksheet, df: pd.DataFrame, workbook, nome_aba: str = "") -> None:
    """
    Escreve um DataFrame em uma worksheet usando write_* direto.
    Mais rápido que pandas.to_excel por evitar o overhead do pandas.
    """
    header_format = workbook.add_format({"bold": True})
    date_format = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})

    # Cabeçalho
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write_string(0, col_idx, str(col_name), header_format)

    # Tipos por coluna (determinados uma vez, não por célula)
    col_types = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            col_types.append("datetime")
        elif pd.api.types.is_bool_dtype(dtype):
            col_types.append("bool")
        elif pd.api.types.is_numeric_dtype(dtype):
            col_types.append("number")
        else:
            col_types.append("string")

    # Dados
    total = len(df)
    for row_idx, row in enumerate(df.itertuples(index=False, name=None), start=1):
        if PROGRESSO_A_CADA and row_idx % PROGRESSO_A_CADA == 0:
            print(f"  [{nome_aba}] {row_idx}/{total} linhas escritas")

        for col_idx, value in enumerate(row):
            ctype = col_types[col_idx]
            # None ou NaN/NaT
            if value is None or value != value:
                worksheet.write_blank(row_idx, col_idx, None)
            elif ctype == "datetime":
                worksheet.write_datetime(row_idx, col_idx, value, date_format)
            elif ctype == "number":
                worksheet.write_number(row_idx, col_idx, value)
            elif ctype == "bool":
                worksheet.write_boolean(row_idx, col_idx, value)
            else:
                worksheet.write_string(row_idx, col_idx, str(value))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    entrada = sys.argv[1] if len(sys.argv) > 1 else ARQUIVO_ENTRADA
    saida = sys.argv[2] if len(sys.argv) > 2 else ARQUIVO_SAIDA

    if not os.path.exists(entrada):
        print(f"[erro] arquivo não encontrado: {entrada}")
        print("Uso: python limpar_precos.py [entrada.xlsx] [saida.xlsx]")
        sys.exit(1)

    print(f"Lendo arquivo: {entrada}")
    abas = pd.read_excel(entrada, sheet_name=None)
    print(f"Abas encontradas: {list(abas.keys())}")

    saidas, estatisticas = separar_por_aba(abas)

    print("\n===== RESUMO CONSOLIDADO =====")
    print(f"Total de linhas lidas:              {estatisticas['total_linhas']}")
    print(f"Total de linhas removidas:          {estatisticas['total_removidas']}")
    print(
        f"  - atenderam ao 1º critério "
        f"(preço <= {LIMITE_PRECO}):        {estatisticas['total_por_preco']}"
    )
    print(
        f"  - atenderam ao 2º critério "
        f"(qtd <= {LIMITE_QUANTIDADE}):          {estatisticas['total_por_quantidade']}"
    )
    print(
        f"  - atenderam ao 3º critério "
        f"(Customer ID vazio):     {estatisticas['total_por_customer_vazio']}"
    )
    print(
        f"  - atenderam ao 4º critério "
        f"(StockCode inválido):    {estatisticas['total_por_stock_code']}"
    )
    print(
        f"  - atenderam aos 4 critérios:       {estatisticas['total_por_ambos']}"
    )
    print(f"Total de linhas limpas:             {estatisticas['total_limpas']}")
    print(f"Total de abas de saída:             {len(saidas)}")
    print("==============================\n")

    print(f"Gravando arquivo de saída: {saida}")
    # XlsxWriter direto + constant_memory: grava em disco conforme gera.
    workbook = xlsxwriter.Workbook(saida, {"constant_memory": True})
    try:
        for nome_aba, df in saidas.items():
            print(f"  -> escrevendo aba '{nome_aba}' ({len(df)} linhas)")
            worksheet = workbook.add_worksheet(nome_aba)
            _escrever_dataframe(worksheet, df, workbook, nome_aba)
    finally:
        workbook.close()

    print(f"[ok] arquivo gerado com as abas: {list(saidas.keys())}")


if __name__ == "__main__":
    main()
