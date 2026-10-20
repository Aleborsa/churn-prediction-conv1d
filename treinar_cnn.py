"""
Fase 3 - Modelo de Machine Learning (CNN 1D) para prever recompra.
Versão Definitiva: Otimização de limiar, baseline justo e EarlyStopping refinado.

Boas práticas aplicadas:
- Transformação log1p no Monetary antes da normalização, para reduzir o
  efeito de outliers (ex.: clientes atacadistas).
- Split 60/20/20 (treino/validação/teste) ANTES da normalização.
- EarlyStopping monitoriza a validação independente com min_delta;
  o teste só é avaliado no final.
- Limiar de decisão otimizado na validação (maximiza F1) para TODOS os
  modelos (CNN, RF e LogReg), garantindo comparação justa.
- Semente de aleatoriedade travada para favorecer a reprodutibilidade.
- Logging em arquivo (pasta logs/) e na consola, em vez de print().

Nota sobre o log1p no Monetary:
- Compare as versões com e sem log1p usando o mesmo split.
- Escolha a transformação com base no desempenho na validação,
  considerando ROC-AUC e AP.
- Depois de selecionar a configuração, avalie-a no conjunto
  de teste reservado, sem usá-lo para novas decisões.
"""

from __future__ import annotations

import io
import logging
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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

# Trava a semente de aleatoriedade (pesos iniciais, shuffle, etc.)
# para garantir reprodutibilidade dos resultados publicados.
tf.keras.utils.set_random_seed(42)

COLUNAS_RFM = ["Recency", "Frequency", "Monetary"]
COL_ALVO = f"Comprou_{DIAS_IGNORADOS}d"


def configurar_logging() -> logging.Logger:
    """
    Configura o logger para escrever em arquivo (pasta logs/) e na consola.

    O nome do arquivo inclui um timestamp para não sobrescrever execuções
    anteriores. Devolve o logger pronto a usar.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"treinar_cnn_{timestamp}.log"

    logger = logging.getLogger("treinar_cnn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging configurado. Arquivo: %s", log_file)
    return logger


logger = configurar_logging()


class LoggingCallback(callbacks.Callback):
    """Registra o progresso de cada época no logger (substitui verbose=1)."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(
            "Época %d - loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f",
            epoch + 1,
            logs.get("loss", 0.0),
            logs.get("accuracy", 0.0),
            logs.get("val_loss", 0.0),
            logs.get("val_accuracy", 0.0),
        )


def imprimir_metricas_validacao(
    nome: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """Exibe métricas de discriminação no conjunto de validação."""
    roc_auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    logger.info(
        "%s — Validação | ROC-AUC: %.4f | AP: %.4f",
        nome,
        roc_auc,
        ap,
    )


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
    Registra ROC-AUC, AP, acurácias (limiar padrão e otimizado), matriz de
    confusão e classification report.

    Recebe as probabilidades contínuas (y_prob) e aplica o limiar para
    gerar as predições binárias. ROC-AUC e AP são calculados sobre as
    probabilidades, independentemente do limiar.

    A acurácia é mostrada em dois valores para deixar explícito o impacto
    do ajuste de limiar (o TensorFlow usa 0.5 internamente; a matriz de
    confusão abaixo reflete o limiar otimizado).
    """
    y_pred = (y_prob >= limiar).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    acc_padrao = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    acc_otim = accuracy_score(y_true, y_pred)

    logger.info("===== %s =====", nome)
    logger.info(
        "ROC-AUC: %.4f | AP: %.4f | Limiar: %.2f", roc_auc, ap, limiar
    )
    logger.info("Acurácia (limiar padrão 0.50): %.4f", acc_padrao)
    logger.info("Acurácia (limiar otimizado %.2f): %.4f", limiar, acc_otim)
    logger.info("Matriz de Confusão:\n%s", confusion_matrix(y_true, y_pred))
    logger.info(
        "Classification Report:\n%s",
        classification_report(
            y_true, y_pred, target_names=["Não comprou (0)", "Comprou (1)"]
        ),
    )


def treinar_baselines(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Treina Random Forest e Regressão Logística como baselines.

    Servem de "termómetro" para comparar com a CNN na tabela do artigo.
    Usam os dados normalizados 2D (sem reshape), pois não são convolucionais.

    Para comparação justa com a CNN, o limiar de decisão de cada baseline
    também é otimizado na validação (maximiza F1) e só depois aplicado no
    teste.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_val_prob = rf.predict_proba(X_val)[:, 1]
    imprimir_metricas_validacao("Random Forest", y_val, rf_val_prob)
    rf_limiar, rf_f1_val = otimizar_limiar(y_val, rf_val_prob)
    logger.info(
        "Limiar ótimo (validação) - Random Forest: %.2f (F1 validação = %.4f)",
        rf_limiar,
        rf_f1_val,
    )
    rf_test_prob = rf.predict_proba(X_test)[:, 1]
    avaliar_modelo("Random Forest (baseline)", y_test, rf_test_prob, limiar=rf_limiar)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_val_prob = lr.predict_proba(X_val)[:, 1]
    imprimir_metricas_validacao("Regressão Logística", y_val, lr_val_prob)
    lr_limiar, lr_f1_val = otimizar_limiar(y_val, lr_val_prob)
    logger.info(
        "Limiar ótimo (validação) - Regressão Logística: %.2f (F1 validação = %.4f)",
        lr_limiar,
        lr_f1_val,
    )
    lr_test_prob = lr.predict_proba(X_test)[:, 1]
    avaliar_modelo(
        "Regressão Logística (baseline)", y_test, lr_test_prob, limiar=lr_limiar
    )


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
    logger.info("Iniciando treino da CNN 1D...")

    caminho = Path(ARQUIVO_SAIDA)
    if not caminho.exists():
        logger.error(
            "Arquivo RFM não encontrado: %s. "
            "Execute engenharia_rfm.py e adicionar_compra_futura.py primeiro.",
            caminho,
        )
        sys.exit(1)

    try:
        X, y = carregar_dados(caminho)
    except (ValueError, FileNotFoundError) as e:
        logger.error("%s", e)
        sys.exit(1)

    # Split 60/20/20 ANTES da normalização (evita data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(X, y)

    # Passo 1 - Normalização (fit apenas no treino)
    X_train_n, X_val_n, X_test_n, _ = normalizar(X_train, X_val, X_test)

    # Baselines (Random Forest e Regressão Logística) em paralelo,
    # com limiar otimizado na validação (mesma regra da CNN)
    treinar_baselines(
        X_train_n, X_val_n, X_test_n, y_train, y_val, y_test
    )

    # Passo 2 - Formatação 3D (amostras, 3, 1)
    X_train_3d = formatar_3d(X_train_n)
    X_val_3d = formatar_3d(X_val_n)
    X_test_3d = formatar_3d(X_test_n)

    # Passo 3 - Construção da CNN
    modelo = construir_cnn(n_passos=X_train_3d.shape[1])

    # Redireciona o summary() (que escreve em stdout) para o log
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        modelo.summary()
    logger.info("Resumo do modelo:\n%s", buffer.getvalue())

    early = callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, restore_best_weights=True
    )

    modelo.fit(
        X_train_3d,
        y_train,
        validation_data=(X_val_3d, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early, LoggingCallback()],
        verbose=0,
    )

    # Otimização do limiar na validação (maximiza F1)
    y_val_prob = modelo.predict(
        X_val_3d, verbose=0
    ).ravel()

    imprimir_metricas_validacao("CNN 1D", y_val, y_val_prob)

    limiar_otimo, f1_val = otimizar_limiar(
        y_val, y_val_prob
    )
    logger.info(
        "Limiar ótimo (validação) - CNN: %.2f (F1 validação = %.4f)",
        limiar_otimo,
        f1_val,
    )

    # Avaliação final no conjunto de teste (nunca visto durante o treino)
    loss, acc = modelo.evaluate(X_test_3d, y_test, verbose=0)
    logger.info(
        "Acurácia no teste (CNN, limiar 0.50 do TF): %.4f | Loss: %.4f",
        acc,
        loss,
    )

    # Métricas académicas da CNN com o limiar ótimo
    y_prob = modelo.predict(X_test_3d, verbose=0).ravel()
    avaliar_modelo("CNN 1D", y_test, y_prob, limiar=limiar_otimo)

    logger.info("Treino concluído.")


if __name__ == "__main__":
    main()
