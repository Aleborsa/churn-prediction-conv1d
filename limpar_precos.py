"""
Remove linhas com preço inválido (<= 0.01) ou quantidade inválida (<= 0)
de um arquivo Excel com múltiplas abas (anos diferentes, mesmas colunas)
e gera um novo Excel mantendo as abas separadas por ano.

Para cada aba de origem, são geradas 2 abas de saída:
    - "limpas_<aba>":    apenas as linhas válidas
                         (Price > 0.01 E Quantity > 0)
    - "removidas_<aba>": apenas as linhas descartadas
                         (Price <= 0.01 OU Quantity <= 0)

Ex.: com 2 abas de entrada -> 4 abas de saída.

A escrita usa o XlsxWriter diretamente (worksheet.write_*) em modo
"constant_memory", que grava as linhas em disco conforme são geradas.
Isso evita o overhead do pandas.to_excel e mantém o uso de memória baixo
mesmo com 1M+ linhas.

Dependência:
    pip install XlsxWriter

Uso:
    python limpar_precos.py
    python limpar_precos.py caminho/para/arquivo.xlsx
    python limpar_precos.py entrada.xlsx saida.xlsx
"""

import os
import re
import sys

import pandas as pd
import xlsxwriter

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
ARQUIVO_ENTRADA = "online_retail_II.xlsx"
ARQUIVO_SAIDA = "online_retail_II_limpo2.xlsx"
COLUNA_PRECO = "Price"          # nome da coluna de preço
LIMITE_PRECO = 0.01             # remove linhas com preço <= este valor
COLUNA_QUANTIDADE = "Quantity"  # nome da coluna de quantidade
LIMITE_QUANTIDADE = 0           # remove linhas com quantidade <= este valor
PREFIXO_REMOVIDAS = "removidas"
PREFIXO_LIMPAS = "limpas"

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
                    - total_por_ambos
                    - total_limpas

    Critérios de remoção (vai para "removidas" se atender QUALQUER um):
        - Price <= LIMITE_PRECO
        - Quantity <= LIMITE_QUANTIDADE
    """
    saidas = {}

    total_linhas = 0
    total_removidas = 0
    total_por_preco = 0
    total_por_quantidade = 0
    total_por_ambos = 0
    total_limpas = 0

    for nome_aba, df in abas.items():
        _validar_coluna_numerica(df, COLUNA_PRECO, nome_aba)
        _validar_coluna_numerica(df, COLUNA_QUANTIDADE, nome_aba)

        # Máscaras individuais (para log)
        mascara_preco = df[COLUNA_PRECO] <= LIMITE_PRECO
        mascara_qtd = df[COLUNA_QUANTIDADE] <= LIMITE_QUANTIDADE

        # Máscara combinada: inválida se preço OU quantidade inválidos
        mascara_invalida = mascara_preco | mascara_qtd

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
        qtd_ambos = int((mascara_preco & mascara_qtd).sum())

        total_linhas += len(df)
        total_removidas += len(removidas)
        total_por_preco += qtd_preco
        total_por_quantidade += qtd_qtd
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
            f"ambos: {qtd_ambos}"
        )

    estatisticas = {
        "total_linhas": total_linhas,
        "total_removidas": total_removidas,
        "total_por_preco": total_por_preco,
        "total_por_quantidade": total_por_quantidade,
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
        f"  - atenderam a AMBOS os critérios:  {estatisticas['total_por_ambos']}"
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
