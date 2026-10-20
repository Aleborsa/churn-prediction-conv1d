"""
Fase 3 - Modelo de Machine Learning (CNN 1D) para prever recompra.

Executa três passos fundamentais:
1. Normalização (MinMaxScaler) das variáveis R, F e M para o intervalo [0, 1].
2. Formatação 3D (samples, steps, variables) para a Conv1D.
3. Construção da CNN (Conv1D + Dense sigmoid) para prever a probabilidade
   de recompra nos DIAS_IGNORADOS dias futuros.

Deve ser executado APÓS adicionar_compra_futura.py, pois depende da coluna
binária 'Comprou_{DIAS_IGNORADOS}d' presente no rfm.xlsx.

Boas práticas aplicadas:
- Transformação log1p no Monetary antes da normalização, para reduzir o
  efeito de outliers (ex.: clientes atacadistas) que comprimiriam a base
  para perto de 0 no MinMaxScaler.
- Split 60/20/20 (treino/validação/teste) ANTES da normalização, para
  evitar data leakage: o MinMaxScaler é ajustado apenas no treino.
- EarlyStopping monitoriza a validação independente; o teste só é
  avaliado no final.
- Formato (amostras, 3, 1) com kernel_size=2: a convolução desliza
  sobre R -> F -> M, justificando o uso de uma CNN.
- Métricas académicas (ROC-AUC, PR-AUC, Precision, Recall, F1) via
  classification_report e roc_auc_score/average_precision_score.
- Baselines (Random Forest e Regressão Logística) treinados em paralelo
  como "termómetros" para comparação na tabela de resultados do artigo.
- Limiar de decisão da CNN otimizado na validação (maximiza F1) e
  aplicado no teste, em vez do corte fixo de 0.5.

Nota sobre o log1p no Monetary (teste A/B sugerido pelo avaliador):
- Execute uma vez com a linha `df["Monetary"] = np.log1p(df["Monetary"])`
  ativa, anote o ROC-AUC.
- Comente essa linha, execute novamente e compare. A versão com melhor
  ROC-AUC é a que deve ir para o artigo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import callbacks, layers, models

from engenharia_rfm import ARQUIVO_SAIDA, DIAS_IGNORADOS, NOME_ABA_SAIDA

COLUNAS_RFM = ["Recency", "Frequency", "Monetary"]
COL_ALVO = f"Comprou_{DIAS_IGNORADOS}d"


def carregar_dados(caminho: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Lê o rfm.xlsx e devolve (X, y) como arrays NumPy.

    Aplica log1p no Monetary para atenuar outliers antes da normalização.
    Para o teste A/B do avaliador, comente a linha do log1p e compare o
    ROC-AUC entre as duas execuções.
    """
    df = pd.read_excel(caminho, sheet_name=NOME_ABA_SAIDA)

    faltantes = [c for c in COLUNAS_RFM + [COL_ALVO] if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Colunas ausentes no RFM: {faltantes}. "
            f"Execute adicionar_compra_futura.py primeiro."
        )

    # Transformação logarítmica no Monetary (reduz o peso de outliers)
    df["Monetary"] = np.log1p(df["Monetary"])

    X = df[COLUNAS_RFM].to_numpy(dtype=float)
    y = df[COL_ALVO].to_numpy(dtype=int)
    return X, y


def dividir_dados(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split 60/20/20 (treino/validação/teste), estratificado pelo alvo.

    Primeiro separa 20% para teste; do restante (80%), separa 25% para
    validação (que corresponde a 20% do total) e fica com 60% para treino.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalizar(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Passo 1: comprime R, F e M para o intervalo [0, 1].

    O scaler é ajustado APENAS no treino (evita data leakage) e a mesma
    régua é aplicada à validação e ao teste.
    """
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_val_norm, X_test_norm, scaler


def formatar_3d(X: np.ndarray) -> np.ndarray:
    """
    Passo 2: converte (amostras, variáveis) -> (amostras, variáveis, 1).

    Cada variável (R, F, M) é tratada como um passo temporal e há 1 canal.
    Com kernel_size=2, a convolução desliza sobre o cliente: primeiro
    cruza Recência com Frequência, depois Frequência com Valor Monetário.
    """
    return X.reshape((X.shape[0], X.shape[1], 1))


def construir_cnn(n_passos: int) -> models.Sequential:
    """
    Passo 3: empilha Conv1D -> Dense -> Dense(sigmoid).

    A Conv1D com kernel_size=2 percorre os 3 passos (R, F, M) extraindo
    as interações entre variáveis adjacentes.
    """
    modelo = models.Sequential(
        [
            layers.Input(shape=(n_passos, 1)),
            layers.Conv1D(
                filters=32, kernel_size=2, activation="relu", padding="same"
            ),
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


def avaliar_modelo(
    nome: str, y_true: np.ndarray, y_prob: np.ndarray, limiar: float = 0.5
) -> None:
    """
    Imprime ROC-AUC, PR-AUC, matriz de confusão e classification report.

    Recebe as probabilidades contínuas (y_prob) e aplica o limiar para
    gerar as predições binárias. ROC-AUC e PR-AUC são calculados sobre
    as probabilidades, independentemente do limiar.
    """
    y_pred = (y_prob >= limiar).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"\n===== {nome} =====")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Limiar: {limiar:.2f}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Não comprou (0)", "Comprou (1)"]
        )
    )


def treinar_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Treina Random Forest e Regressão Logística como baselines.

    Servem de "termómetro" para comparar com a CNN na tabela do artigo.
    Usam os dados normalizados 2D (sem reshape), pois não são convolucionais.
    Enviam as probabilidades (predict_proba[:, 1]) para avaliar_modelo.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    avaliar_modelo("Random Forest (baseline)", y_test, rf_prob)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    avaliar_modelo("Regressão Logística (baseline)", y_test, lr_prob)


def otimizar_limiar(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Procura o limiar em [0.1, 0.9) (passo 0.05) que maximiza o F1-Score
    no conjunto de validação. Devolve (melhor_limiar, melhor_f1).
    """
    melhor_limiar = 0.5
    melhor_f1 = -1.0
    for limiar in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= limiar).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_limiar = float(limiar)
    return melhor_limiar, melhor_f1


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

    # Split 60/20/20 ANTES da normalização (evita data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(X, y)

    # Passo 1 - Normalização (fit apenas no treino)
    X_train_n, X_val_n, X_test_n, _ = normalizar(X_train, X_val, X_test)

    # Baselines (Random Forest e Regressão Logística) em paralelo
    treinar_baselines(X_train_n, X_test_n, y_train, y_test)

    # Passo 2 - Formatação 3D (amostras, 3, 1)
    X_train_3d = formatar_3d(X_train_n)
    X_val_3d = formatar_3d(X_val_n)
    X_test_3d = formatar_3d(X_test_n)

    # Passo 3 - Construção da CNN
    modelo = construir_cnn(n_passos=X_train_3d.shape[1])
    modelo.summary()

    early = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    modelo.fit(
        X_train_3d,
        y_train,
        validation_data=(X_val_3d, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early],
        verbose=1,
    )

    # Otimização do limiar na validação (maximiza F1)
    y_val_prob = modelo.predict(X_val_3d, verbose=0).ravel()
    limiar_otimo, f1_val = otimizar_limiar(y_val, y_val_prob)
    print(
        f"\nLimiar ótimo (validação): {limiar_otimo:.2f} "
        f"(F1 validação = {f1_val:.4f})"
    )

    # Avaliação final no conjunto de teste (nunca visto durante o treino)
    loss, acc = modelo.evaluate(X_test_3d, y_test, verbose=0)
    print(f"\nAcurácia no teste (CNN): {acc:.4f} | Loss: {loss:.4f}")

    # Métricas académicas da CNN com o limiar ótimo
    y_prob = modelo.predict(X_test_3d, verbose=0).ravel()
    avaliar_modelo("CNN 1D", y_test, y_prob, limiar=limiar_otimo)


if __name__ == "__main__":
    main()
