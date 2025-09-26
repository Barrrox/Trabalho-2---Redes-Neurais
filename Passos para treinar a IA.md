# Rede Neural de Categorização de Movimento Artístico

Este repositório contém o código para uma rede neural convolucional (CNN) projetada para classificar pinturas com base em seus respectivos movimentos artísticos.

## Requisitos

Antes de iniciar o treinamento, certifique-se de que você tenha todas as bibliotecas necessárias instaladas. Você pode instalá-las usando o `pip`.

```bash
pip install tensorflow numpy opencv-python scikit-learn matplotlib
```

## Passo a Passo para Treinar o Modelo

### 1. Baixar a Base de Dados
A base de dados com as imagens das pinturas está no Google Drive. Faça o download e extraia o conteúdo para a pasta raiz do projeto.


### 2. Pré-processar as Imagens
Execute o script `megaformat.py` para formatar e preparar as imagens para o treinamento. Este script irá redimensionar, normalizar e organizar os dados em um formato que a rede possa consumir.

* **Aviso:** Este processo demorou cerca de 8 minutos no meu computador (Processador, 12th Gen Intel(R) Core(TM) i5-1235U, 1300 Mhz, 10 Núcleo(s), 12 Processador(es) Lógico(s)). Deve ser mais rápido no seu. 

```bash
python megaformat.py
```

### 3. Executar o Treinamento

* **Aviso:** Demora um pouco... Na verdade esse processo nem rodou no meu computador com as 9k imagens por falta de memória RAM (tinha 8GB no total usando windowns).

```bash
python treinamento.py
```

### 4. Verificar os Resultados

```bash
python carregar_modelo.py
```
