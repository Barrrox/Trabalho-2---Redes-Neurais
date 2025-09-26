# -*- coding-utf-8 -*-

"""
Classificação de Obras de Arte por Movimento Artístico

Script para treinar uma rede neural convolucional (baseada na arquitetura Inception)
para classificar imagens de obras de arte em 9 movimentos artísticos distintos.

Autores: Ellen Brzozoski, João Silva, Lóra, Matheus Barros
"""

# --- Importação das Bibliotecas ---

from numpy import load, array 
# Ferramentas do TensorFlow/Keras pra construir e treinar a rede neural.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Concatenate, Dropout
from tensorflow.keras.utils import to_categorical
# Sklearn ajuda a dividir os dados de forma organizada.
from sklearn.model_selection import train_test_split
# Módulo pra medir o tempo de execução de cada etapa.
from time import time


def treinar_modelo(TAM_TESTES, QNT_EPOCAS):
    """
    Orquestra todo o processo de treinamento da rede neural: carrega os dados,
    pré-processa as imagens, define a arquitetura do modelo, compila, treina
    e salva o resultado final.

    Args:
        TAM_TESTES (float): Proporção do dataset a ser usada para teste (ex: 0.15 para 15%).
        QNT_EPOCAS (int): Número de vezes que o modelo verá o dataset de treino completo.
    """
    
    inicio = time()

    # --- 1. Carregamento e Preparação dos Dados ---

    # Carrega os datasets pré-processados do disco.
    # Espera-se que 'imagens_treino.npy' contenha todas as imagens como arrays (128, 128, 3)
    # e 'labels_treino.npy' contenha os rótulos numéricos correspondentes.
    imagens = load("imagens_treino.npy")
    labels = load("labels_treino.npy")

    print(f"Tempo = {time() - inicio:.2f}s : Datasets carregados da memória.")

    # Divide o conjunto de dados em treino e teste.
    x_train, x_test, y_train, y_test = train_test_split(
        imagens, 
        labels, 
        test_size=TAM_TESTES, 
        # random_state=42 garante que a divisão seja sempre a mesma.
        random_state=42 
    )
    
    print(f"Tempo = {time() - inicio:.2f}s : Dados de treino e teste separados.")

    # Normaliza os valores dos pixels das imagens.
    # As imagens vêm com pixels de 0 a 255. Dividir por 255.0 coloca tudo
    # no intervalo [0, 1]. Isso ajuda o algoritmo de otimização (Adam)
    # a convergir mais rápido e de forma mais estável.
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(f"Tempo = {time() - inicio:.2f}s : Pixels normalizados entre 0 e 1.")

    # Converte os rótulos (labels) para o formato "one-hot encoding".
    # Ex: a label '3' vira um vetor [0, 0, 0, 1, 0, 0, 0, 0, 0].
    # A rede neural usa esse formato na camada de saída com ativação softmax.
    # O número 9 representa a quantidade de classes (movimentos artísticos) do nosso problema.
    y_train = to_categorical(y_train, 9)
    y_test = to_categorical(y_test, 9)

    print(f"Tempo = {time() - inicio:.2f}s : Rótulos convertidos para o formato one-hot.")

    # --- 2. Definição da Arquitetura do Modelo (GoogLeNet/Inception modificado) ---

    def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
        """
        Define um bloco de construção "Inception", que aplica filtros de diferentes 
        tamanhos na mesma entrada e depois concatena tudo. A ideia é capturar padrões
        em diferentes escalas (detalhes pequenos com 1x1, maiores com 3x3 e 5x5). 

        Args:
            x (Tensor): Tensor de entrada para o módulo.
            filters_... (int): Número de filtros para cada caminho convolucional.
        
        Returns:
            Tensor: Tensor de saída com as features concatenadas.
        """
        # Ramo 1: Convolução 1x1. Captura padrões bem localizados.
        conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

        # Ramo 2: Convolução 3x3.
        # A conv 1x1 anterior reduz a profundidade do tensor (dimensão dos filtros),
        # o que diminui drasticamente o custo computacional da conv 3x3 seguinte.
        conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
        conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)

        # Ramo 3: Convolução 5x5. Mesma lógica do ramo 2.
        conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
        conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)

        # Ramo 4: Max Pooling seguido de uma convolução 1x1.
        pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_proj_conv = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

        # Junta os resultados de todos os ramos em um único tensor.
        # O 'axis=-1' significa que estamos empilhando na dimensão dos filtros.
        output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool_proj_conv])

        return output

    # -- Construção do Modelo --
    
    # Camada de entrada: define o formato que a rede espera receber.
    # No nosso caso, imagens de 128x128 pixels com 3 canais de cor (RGB).
    input_layer = Input(shape=(128, 128, 3))

    # Bloco convolucional inicial pra uma primeira extração de features
    # e redução da dimensionalidade da imagem.
    x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(input_layer)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    # Primeiro módulo Inception.
    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32)

    # Segundo módulo Inception, com mais filtros para aprender padrões mais complexos.
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    # Bloco de Classificação: prepara a saída dos módulos para a decisão final.
    x = Flatten()(x) # Transforma o mapa de features 2D em um vetor 1D.
    x = Dense(128, activation='relu')(x) # Camada densa, o "cérebro" do classificador.
    
    # Dropout é uma técnica de regularização para evitar overfitting.
    # Durante o treino, 50% dos neurônios da camada anterior são "desligados"
    # aleatoriamente a cada passo. Isso força a rede a não depender de neurônios específicos.
    x = Dropout(0.5)(x)
    
    # Camada de saída final.
    # 9 neurônios, um pra cada movimento artístico.
    # 'softmax' transforma a saída em um vetor de probabilidades, onde a soma de tudo é 1.
    output_layer = Dense(9, activation='softmax')(x)

    # Monta o modelo final, conectando a entrada com a saída.
    model = Model(inputs=input_layer, outputs=output_layer, name='inception_style_model')

    # Imprime um resumo da arquitetura da rede. Ótimo pra debugar.
    model.summary()

    # --- 3. Compilação e Treinamento ---

    # Configura o modelo para o treinamento.
    # 'adam' é um otimizador eficiente e muito usado.
    # 'categorical_crossentropy' é a função de perda padrão para classificação multiclasse.
    # 'accuracy' é a métrica que queremos observar (percentual de acertos).
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Tempo = {time() - inicio:.2f}s : Modelo compilado e pronto para treinar.")

    # Inicia o treinamento do modelo.
    # O 'validation_data' é usado ao final de cada época para avaliar o desempenho
    # do modelo em dados que ele não usou para treinar, ajudando a monitorar o overfitting.
    model.fit(x_train, y_train, epochs=QNT_EPOCAS, validation_data=(x_test, y_test))

    print(f"Tempo = {time() - inicio:.2f}s : Modelo treinado com sucesso.")

    # Salva o modelo treinado em um único arquivo.
    model.save('model.keras')

    print(f"Tempo = {time() - inicio:.2f}s : Modelo salvo como 'model.keras'.")


def main():
    """
    Função principal que define os hiperparâmetros e chama a função de treinamento.
    """
    # Define a proporção do dataset a ser usada como conjunto de teste.
    # 15% dos dados serão usados para validar o modelo.
    TAM_TESTES = 15/100

    # Define o número de épocas.
    # Uma época é um ciclo completo de treinamento sobre todo o dataset.
    QNT_EPOCAS = 40

    treinar_modelo(TAM_TESTES, QNT_EPOCAS)


main()