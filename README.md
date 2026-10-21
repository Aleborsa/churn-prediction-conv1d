# Predição de Recompra (Churn) com Redes Neurais Convolucionais 1D

Este repositório contém o código-fonte e a metodologia para a predição de comportamento de clientes utilizando a matriz RFM (Recência, Frequência e Valor Monetário). O projeto foca na implementação de uma **Rede Neural Convolucional 1D (Conv1D)** desenvolvida em TensorFlow/Keras, comparando o seu desempenho com baselines clássicos (Random Forest e Regressão Logística).

## 🧠 Arquitetura e Decisões Metodológicas

Para fins de avaliação acadêmica, os seguintes destaques técnicos foram implementados na pipeline:

* **Prevenção de Data Leakage:** Separação estrita de conjuntos de Treino (60%), Validação (20%) e Teste (20%). O `MinMaxScaler` foi ajustado (`fit`) exclusivamente no Treino.
* **Modelagem Espacial (Conv1D):** Conversão da matriz 2D tabular para tensores 3D. Utilização de `kernel_size=2` para extração de correlações adjacentes iterativas (Recência + Frequência, seguido de Frequência + Valor Monetário).
* **Otimização de Limiar (Threshold Dinâmico):** Abandono do limiar padrão (0.50). O limiar de decisão foi dinamicamente calibrado no conjunto de validação para maximizar o F1-Score, equilibrando Precision e Recall.
* **Governança e Rastreabilidade:** Implementação de callbacks para Early Stopping (prevenção de overfitting) e logs detalhados salvos automaticamente em disco.

---

## ⚙️ Pré-requisitos e Configuração do Ambiente

Este projeto foi desenvolvido e homologado utilizando o **Python 3.13**.
Recomenda-se fortemente a criação de um ambiente virtual para isolar as dependências e evitar conflitos.

### 1. Estrutura de Pastas Esperada
Certifique-se de que o diretório do projeto esteja organizado da seguinte forma antes de executar os scripts. O arquivo bruto `online_retail_II.xlsx` **deve** estar dentro da pasta `dados/`.

```text
meu_projeto/
├── dados/
│   └── online_retail_II.xlsx      <-- (Coloque a base bruta aqui)
├── logs/
├── src/
│   ├── limpeza_dados/
│   │   └── limpeza_dados.py
│   ├── eda/
│   │   └── eda.py
│   └── cnn/
│       ├── engenharia_rfm.py
│       ├── adicionar_compra_futura.py
│       └── treinar_cnn.py
├── requirements.txt
└── README.md
```

### 2. Instalar as dependências

Abra o terminal na pasta raiz do projeto e instale as bibliotecas utilizando o arquivo fornecido (incluindo TensorFlow 2.21.0, Pandas 3.0.6, Scikit-Learn 1.9.1, etc.):

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar o Pipeline (Passo a Passo)

O projeto foi desenhado como um pipeline de dados sequencial. Execute os scripts na exata ordem abaixo, a partir da pasta raiz do projeto:

### Passo 0: Análise Exploratória (Opcional)
Gera um relatório descritivo sobre a saúde da base original.
```bash
python src/eda/eda.py
```
* **Saída:** Um arquivo `relatorio_eda.txt` será gerado na pasta `dados/`.

### Passo 1: Limpeza e Higienização
Remove anomalias financeiras (preços negativos, devoluções) e clientes não identificados.
```bash
python src/limpeza_dados/limpeza_dados.py
```
* **Saída:** O arquivo `online_retail_II_limpo.xlsx` será gerado na pasta `dados/`.

### Passo 2: Engenharia de Atributos (Matriz RFM)
Converte as transações de vendas na matriz RFM agregada por cliente. Este script exclui os últimos 90 dias da base no cálculo e, automaticamente, executa o script `adicionar_compra_futura.py` para gerar o "Gabarito" (Target) na janela separada.
```bash
python src/cnn/engenharia_rfm.py
```
* **Saída:** O arquivo final `rfm.xlsx` contendo as features (R, F, M) e a variável alvo binária será gerado na pasta `dados/`.

### Passo 3: Treinamento da Inteligência Artificial
Carrega a matriz RFM consolidada, realiza a divisão estratificada (Treino/Validação/Teste), aplica a normalização `MinMaxScaler`, converte para tensores 3D e treina a Rede Neural e os modelos de baseline.
```bash
python src/cnn/treinar_cnn.py
```
* **O que esperar:** 
  1. O console exibirá o progresso dos logs e as métricas.
  2. Um diretório `logs/` será criado automaticamente na raiz, contendo um arquivo completo de rastreabilidade (ex: `treinar_cnn_YYYYMMDD_HHMMSS.log`) documentando toda a avaliação matemática da rede.

