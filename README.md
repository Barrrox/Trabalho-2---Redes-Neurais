# Rede Neural de Categorização de Movimento Artístico

Este repositório contém o código para uma rede neural convolucional (CNN) projetada para classificar pinturas com base em seus respectivos movimentos artísticos.

## Requisitos

Antes de iniciar o treinamento, certifique-se de que você tenha todas as bibliotecas necessárias instaladas. Você pode instalá-las usando o `pip`.

* **TensorFlow:** A principal biblioteca para a construção e treinamento da rede neural.
    ```bash
    pip install tensorflow
    ```

* **NumPy:** Essencial para manipulação de arrays e operações matemáticas.
    ```bash
    pip install numpy
    ```

* **OpenCV (cv2):** Utilizada para o processamento e leitura das imagens.
    ```bash
    pip install opencv-python
    ```

* **Scikit-learn:** Usada para gerar a matriz de confusão e avaliar a performance do modelo.
    ```bash
    pip install scikit-learn
    ```

* **Matplotlib:** Necessária para a visualização da matriz de confusão e outros gráficos.
    ```bash
    pip install matplotlib
    ```

## Passo a Passo para Treinar o Modelo

### 1. Baixar a Base de Dados
A base de dados com as imagens das pinturas está no Google Drive. Faça o download e extraia o conteúdo para a pasta raiz do projeto.


### 2. Pré-processar as Imagens
Execute o script `megaformat.py` para formatar e preparar as imagens para o treinamento. Este script irá redimensionar, normalizar e organizar os dados em um formato que a rede possa consumir.

* **Aviso:** Este processo demorou cerca de 8 minutos no meu computador. Deve ser mais rápido no seu.

```bash
python megaformat.py
```

### 3. Executar o Treinamento

```bash
python treinamento.py
```

### 4. Verificar os Resultados

```bash
python carregar_modelo.py
```
