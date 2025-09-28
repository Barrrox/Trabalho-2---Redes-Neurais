# -*- coding: utf-8 -*-
"""
Este script realiza a otimização de hiperparâmetros para uma rede neural convolucional,
baseada na arquitetura Inception, para classificar obras de arte por movimento artístico.
A busca pelos melhores parâmetros é feita com RandomizedSearchCV sobre o conjunto de treino.

Autores: Ellen Brzozoski, João Silva, Lóra, Matheus Barros
"""

import numpy as np
from time import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Input,
    Concatenate, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Activation
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

def otimizar_parametros(TAM_TESTE, TAM_VALIDACAO, FRACAO_BUSCA):
    """
    Carrega os dados, divide-os em treino/validação/teste e executa o
    RandomizedSearchCV para encontrar os melhores hiperparâmetros para o modelo.
    
    Args:
        TAM_TESTE (float): A proporção do conjunto de dados a ser usada para o teste.
        TAM_VALIDACAO (float): A proporção do conjunto de dados a ser usada para a validação.
        FRACAO_BUSCA (float): A proporção do conjunto de treino a ser usada na busca (ex: 0.5 para 50%).
    """
    inicio = time()

    # Carrega o conjunto de dados de treino completo.
    imagens = np.load("imagens_treino.npy")
    labels = np.load("labels_treino.npy")

    print(f"Tempo = {time() - inicio:.2f}s : Dados de treino carregados")

    # Separa o conjunto de teste (ex: 15%) do restante (ex: 85%).
    x_train_temp, x_test, y_train_temp, y_test = train_test_split(
        imagens, labels, test_size=TAM_TESTE, random_state=42, stratify=labels
    )

    # Separa o restante em treino e validação.
    val_size_recalculado = TAM_VALIDACAO / (1.0 - TAM_TESTE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_temp, y_train_temp, test_size=val_size_recalculado, random_state=42, stratify=y_train_temp
    )

    print(f"Tempo = {time() - inicio:.2f}s : Dados divididos em treino, validação e teste")
    print(f"Conjunto de Treino Completo: {len(x_train)} amostras")
    print(f"Conjunto de Validação: {len(x_val)} amostras")
    print(f"Conjunto de Teste: {len(x_test)} amostras")

    # Se FRACAO_BUSCA for menor que 1.0, reduz o conjunto de treino para a busca
    if FRACAO_BUSCA < 1.0 and FRACAO_BUSCA > 0.0:
        # Usamos train_test_split para pegar uma amostra estratificada. O restante é descartado para esta busca.
        # y_train ainda não está em one-hot, o que é ideal para o parâmetro stratify.
        _, x_train, _, y_train = train_test_split(
            x_train, y_train,
            test_size=FRACAO_BUSCA,
            random_state=42,
            stratify=y_train
        )
        print(f"Tempo = {time() - inicio:.2f}s : Subconjunto de treino criado para a busca.")

    print(f"Conjunto de Treino para a Busca: {len(x_train)} amostras ({FRACAO_BUSCA*100:.0f}%)")

    # Normaliza os valores dos pixels do conjunto de treino para o intervalo [0, 1].
    x_train = x_train / 255.0

    print(f"Tempo = {time() - inicio:.2f}s : Pixels normalizados entre 0 e 1")

    # Converte os rótulos do conjunto de treino para o formato one-hot encoding (categórico).
    y_train = to_categorical(y_train, 9)

    print(f"Tempo = {time() - inicio:.2f}s : Rótulos convertidos para categorias")

    def construir_modelo_inception(optimizer='adam', dense_units=128):
        """
        Constrói e compila o modelo Keras otimizado com arquitetura Inception.
        Esta função é parametrizada para ser usada pelo RandomizedSearchCV.

            Args:
                optimizer (str): O otimizador a ser usado na compilação.
                dense_units (int): O número de neurônios na penúltima camada densa.

            Returns:
                Model: O modelo Keras compilado.
        """
        def inception_module(x, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
            # Ramo 1
            conv1 = Conv2D(f1, (1,1), padding='same', use_bias=False)(x)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)

            # Ramo 2
            conv3_in = Conv2D(f2_in, (1,1), padding='same', use_bias=False)(x)
            conv3_in = BatchNormalization()(conv3_in)
            conv3_in = Activation('relu')(conv3_in)
            conv3 = Conv2D(f2_out, (3,3), padding='same', use_bias=False)(conv3_in)
            conv3 = BatchNormalization()(conv3)
            conv3 = Activation('relu')(conv3)

            # Ramo 3
            conv5_in = Conv2D(f3_in, (1,1), padding='same', use_bias=False)(x)
            conv5_in = BatchNormalization()(conv5_in)
            conv5_in = Activation('relu')(conv5_in)
            conv5 = Conv2D(f3_out, (5,5), padding='same', use_bias=False)(conv5_in)
            conv5 = BatchNormalization()(conv5)
            conv5 = Activation('relu')(conv5)

            # Ramo 4
            pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
            pool_proj = Conv2D(f4_out, (1,1), padding='same', use_bias=False)(pool)
            pool_proj = BatchNormalization()(pool_proj)
            pool_proj = Activation('relu')(pool_proj)

            out = Concatenate(axis=-1)([conv1, conv3, conv5, pool_proj])
            return out

        input_layer = Input(shape=(128, 128, 3))


        # Convolução: Analisa a imagem com 32 neuronios (filtros)
        # de tamanho 3x3 para fazer uma captura inicial das imagens, 
        # saltando de 2 em 2 pixels. Transforma (128,128,3) para
        # (64, 64, 32). A seguir há um BatchNormalization que contém 
        # um parâmetro beta, o qual aje como um bias. Logo é possível 
        # desativar o bias aqui para otimizar um pouco o treinamento.    
        x = Conv2D(32, (3,3), strides=(2,2), padding='same', use_bias=False)(input_layer)
        
        # BatchNormalization para padronizar a saída convolucional.
        # Ele calcula a média, desvio padrão e normaliza os dados.
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # MaxPooling2D reduz a dimensionalidade da imagem ao
        # analisar porçoes 2x2 da imagem e capturar o pixel com
        # maior valor (Max) formando um novo tensor com metade
        # do tamanho. strides=(2,2) indica que a analise é feita
        # saltando de 2 em 2 pixels, o que reduz a imagem a sua
        # metade.
        x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

        # Os dois primeiros módulos inception do GoogleLeNet
        x = inception_module(x, 64, 96, 128, 16, 32, 32)
        x = inception_module(x, 128, 128, 192, 32, 96, 64)

        # Explicação igual ao maxpooling anterior
        x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

        # Global Average Pooling faz uma média dos mapas de características
        # para cada canal (transforma o tensor3D anterior de 16x16x480 
        # em um tensor 1D de tamanho 480). Serve para reduzir o tamanho do mapa de
        # características para a próxima camada.
        x = GlobalAveragePooling2D()(x)

        # Camada totalmente conectada com a anteior para aprender combinações complexas
        # de caracteristicas.
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Camada final que transforma os 256 neurônios anteriores
        # em 9, para obter o resultado da época. Utiliza o softmax
        # para trasnformar os resultados em um vetor de probabilidades.

        output_layer = Dense(9, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    # "Empacota" o modelo Keras para que seja compatível com a API do scikit-learn.
    model = KerasClassifier(model=construir_modelo_inception, verbose=2)

    print(f"Tempo = {time() - inicio:.2f}s : Modelo empacotado com KerasClassifier")

    # Define o espaço de busca dos hiperparâmetros a serem testados.
    parametros = {
        # Parâmetros do método .fit()
        'batch_size': [16, 32, 64],
        'epochs': [50],  # Aumentamos as épocas, pois o EarlyStopping cuidará do limite
        'validation_split': [0.2], # Separa 20% dos dados de treino para validação em cada fold

        # Parâmetros da função 'construir_modelo_inception' (prefixo 'model__')
        'model__optimizer': ['adam', 'rmsprop'],
        'model__dense_units': [128, 256, 512],
    }

    # Configura a busca aleatória com validação cruzada (cv=3).
    random_search = RandomizedSearchCV(estimator=model, param_distributions=parametros,
                                         n_iter=6, cv=2, verbose=2)

    print(f"Tempo = {time() - inicio:.2f}s : Instância do RandomizedSearch criada")

    # Cria o callback de Early Stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=2,
        restore_best_weights=True
    )

    # Inicia o processo de busca, passando o callback para o método .fit()
    modelo_convergido = random_search.fit(x_train, y_train, callbacks=[early_stop])

    print(f"Tempo = {time() - inicio:.2f}s : Otimização de hiperparâmetros concluída")

    # Exibe os melhores resultados encontrados.
    print(f"\nMelhores parâmetros encontrados: {modelo_convergido.best_params_}")
    print(f"Melhor score de validação cruzada: {modelo_convergido.best_score_:.4f}")


def main():
    """
    Ponto de entrada do script. Define a proporção do dataset a ser usada
    e chama a função de otimização.
    """
    # Define a proporção para os conjuntos de teste e validação (ex: 15% cada).
    # O restante será usado para o treino (ex: 70%).
    TAM_TESTE = 0.10
    TAM_VALIDACAO = 0.10
    
    # Novo parâmetro: Define a fração do conjunto de treino que será usada para a busca.
    # Por exemplo, 0.5 usará 50% dos dados de treino para acelerar a otimização.
    # Para usar todos os dados de treino, defina como 1.0.
    FRACAO_BUSCA = 0.20
    
    otimizar_parametros(TAM_TESTE, TAM_VALIDACAO, FRACAO_BUSCA)

# Bloco padrão para garantir que main() seja executado apenas quando o script for chamado diretamente.
if __name__ == "__main__":
    main()