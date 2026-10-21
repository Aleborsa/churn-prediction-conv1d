# Predição de Recompra no Varejo Online (CNN 1D)

## Como replicar o ambiente
1. Clone o repositório.
2. Crie um ambiente virtual: `python -m venv .venv`
3. Ative o ambiente e instale as dependências: `pip install -r requirements.txt`
4. Coloque o arquivo `online_retail_II.xlsx` dentro da pasta `/dados`.

## Ordem de Execução
Execute os scripts a partir da raiz do projeto na seguinte ordem:
1. `python src/limpeza_dados/limpar_precos.py`
2. `python src/cnn/engenharia_rfm.py`
3. `python src/cnn/adicionar_compra_futura.py`
4. `python src/cnn/treinar_cnn.py`