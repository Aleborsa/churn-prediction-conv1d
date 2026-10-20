"""
Fase 3 - Modelo de Machine Learning (CNN 1D) para prever recompra.

Executa três passos fundamentais:
1. Normalização (MinMaxScaler) das variáveis R, F e M para o intervalo [0, 1].
2. Formatação 3D (samples, steps, variables) para a Conv1D.
3. Construção da CNN (Conv1D + Dense sigmoid) para prever a probabilidade
   de recompra nos DIAS_IGNORADOS dias futuros.

Deve ser executado APÓS adicionar_compra_futura.py, pois depende da coluna
binária 'Comprou_{DIAS_IGNORADOS}d' presente no rfm.xlsx.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import callbacks, layers, models

from engenharia_rfm import ARQUIVO_SAIDA, DIAS_IGNORADOS, NOME_ABA_SAIDA

COLUNAS_RFM = ["Recency", "Frequency", "Monetary"]
COL_ALVO = f"Comprou_{DIAS_IGNORADOS}d"


def carregar_dados(caminho: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê o rfm.xlsx e devolve (X, y) como arrays NumPy."""
    df = pd.read_excel(caminho, sheet_name=NOME_ABA_SAIDA)

    faltantes = [c for c in COLUNAS_RFM + [COL_ALVO] if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Colunas ausentes no RFM: {faltantes}. "
            f"Execute adicionar_compra_futura.py primeiro."
        )

    X = df[COLUNAS_RFM].to_numpy(dtype=float)
    y = df[COL_ALVO].to_numpy(dtype=int)
    return X, y


def normalizar(X: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Passo 1: comprime R, F e M para o intervalo [0, 1]."""
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler


def formatar_3d(X: np.ndarray) -> np.ndarray:
    """
    Passo 2: converte (amostras, variáveis) -> (amostras, passos, canais).
    Cada variável (R, F, M) é tratada como um passo temporal e há 1 canal.
    """
    return X.reshape((X.shape[0], X.shape[1], 1))


def construir_cnn(n_passos: int) -> models.Sequential:
    """Passo 3: empilha Conv1D -> Dense -> Dense(sigmoid)."""
    modelo = models.Sequential(
        [
            layers.Input(shape=(n_passos, 1)),
            layers.Conv1D(
                filters=32, kernel_size=2, activation="relu", padding="same"
            ),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def main() -> None:
    caminho = Path(ARQUIVO_SAIDA)
    if not caminho.exists():
        print(
            f"[ERRO] Arquivo RFM não encontrado: {caminho}. "
            f"Execute engenharia_rfm.py e adicionar_compra_futura.py primeiro.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        X, y = carregar_dados(caminho)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)

    # Passo 1 - Normalização
    X_norm, _ = normalizar(X)

    # Passo 2 - Formatação 3D
    X_3d = formatar_3d(X_norm)

    # Split treino/teste (estratificado para preservar a proporção de 0/1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_3d, y, test_size=0.2, random_state=42, stratify=y
    )

    # Passo 3 - Construção da CNN
    modelo = construir_cnn(n_passos=X_3d.shape[1])
    modelo.summary()

    early = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    modelo.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early],
        verbose=1,
    )

    loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia no teste: {acc:.4f} | Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
