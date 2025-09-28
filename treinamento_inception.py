# -*- coding: utf-8 -*-

"""
Treinamento otimizado de CNN estilo Inception
para classificar pinturas por movimento artístico.
"""

import numpy as np
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, Conv2D, MaxPooling2D, Input,
    Concatenate, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Activation
) 
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from sklearn.model_selection import train_test_split
from time import time


def treinar_modelo(TAM_TESTES, TAM_VALIDACAO, QNT_EPOCAS):
    inicio = time()

    # --- 1. Carregamento dos Dados ---
    imagens = np.load("imagens_treino.npy")
    labels = np.load("labels_treino.npy")
    print(f"Tempo = {time() - inicio:.2f}s : Datasets carregados.")

    # Divisão em treino, validação e teste
    # Passo 1: Separar o conjunto de teste
    x_train_temp, x_test, y_train_temp, y_test = train_test_split(
        imagens, labels, test_size=TAM_TESTES, random_state=42, stratify=labels
    )

    # Passo 2: Separar o restante em treino
    val_size_recalculado = TAM_VALIDACAO / (1.0 - TAM_TESTES)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_temp, y_train_temp, test_size=val_size_recalculado, random_state=42, stratify=y_train_temp
    )

    # Normalização
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0
    print(f"Tempo = {time() - inicio:.2f}s : Normalização concluída.")

    # One-hot encoding
    y_train = to_categorical(y_train, 9)
    y_val = to_categorical(y_val, 9)
    y_test = to_categorical(y_test, 9)

    # --- 2. Data Augmentation ---
    # Modifica as imagens para 
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    datagen.fit(x_train)

    # --- 3. Definição do Modelo ---

    # Módulo Inception 
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
    x = Conv2D(32, (3,3), strides=(2,2), padding='same', use_bias=False)(input_layer) # use_bias=False (ver ponto 3)
    
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
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) # DropOut alto para lidar com a camada densa.

    # Camada final que transforma os 256 neurônios anteriores
    # em 9, para obter o resultado da época. Utiliza o softmax
    # para trasnformar os resultados em um vetor de probabilidades.
    output_layer = Dense(9, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # --- 4. Callbacks ---
    
    # EarlyStopping para o treinamento se não houver melhora na taxa
    # de erro após 15 épocas seguidas
    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    # Salva o melhor modelo baseando-se no valor da acurácia de validação
    checkpoint = ModelCheckpoint(
        "modelo.keras", monitor="val_accuracy", save_best_only=True
    )

    # --- 5. Treinamento ---
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=16),
        validation_data=(x_val, y_val),
        epochs=QNT_EPOCAS,
        callbacks=[early_stop, checkpoint]
    )

    print(f"Tempo total = {time() - inicio:.2f}s : Treinamento finalizado.")
    
    # Avaliação final e imparcial no conjunto de teste
    print("\n--- Avaliando o melhor modelo no conjunto de teste ---")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%")

    model.save("modelo_final.keras")
    print("Modelos salvos: 'best_model.keras' (melhor) e 'final_model.keras' (última época).")
    return history


def main():
    # Proporção 80% treino, 10% validação, 10% teste
    TAM_TESTES = 0.10
    TAM_VALIDACAO = 0.10
    QNT_EPOCAS = 60
    treinar_modelo(TAM_TESTES, TAM_VALIDACAO, QNT_EPOCAS)


if __name__ == "__main__":
    main()