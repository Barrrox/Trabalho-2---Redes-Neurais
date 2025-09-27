# -*- coding: utf-8 -*-

"""
Treinamento otimizado de CNN estilo Inception
para classificar pinturas por movimento artístico.
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Input,
    Concatenate, Dropout, BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from time import time


def treinar_modelo(TAM_TESTES, TAM_VALIDACAO, QNT_EPOCAS):
    inicio = time()

    # --- 1. Carregamento dos Dados ---
    imagens = np.load("imagens_treino.npy")
    labels = np.load("labels_treino.npy")
    print(f"Tempo = {time() - inicio:.2f}s : Datasets carregados.")

    # Divisão em treino, validação e teste
    # Passo 1: Separar o conjunto de teste (10%) do restante (90%)
    x_train_temp, x_test, y_train_temp, y_test = train_test_split(
        imagens, labels, test_size=TAM_TESTES, random_state=42, stratify=labels
    )

    # Passo 2: Separar o restante em treino (80%) e validação (10%)
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
    def inception_module(x, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(x)

        conv3_in = Conv2D(f2_in, (1,1), padding='same', activation='relu')(x)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3_in)

        conv5_in = Conv2D(f3_in, (1,1), padding='same', activation='relu')(x)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5_in)

        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
        pool_proj = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)

        out = Concatenate(axis=-1)([conv1, conv3, conv5, pool_proj])
        out = BatchNormalization()(out)
        return out

    input_layer = Input(shape=(128, 128, 3))

    # Dropout leve para desligar 20% dos neurônios da camada inicial.
    x = Dropout(0.2)(input_layer)

    x = Conv2D(32, (3,3), activation='relu', strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    # Dropout para desligar 30% dos neurônios da camada inception.
    # Isso ajuda a evitar dependência de certos neur
    x = Dropout(0.3)(x) 

    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = Dropout(0.3)(x)

    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) # DropOut alto para lidar com as camadas Densas

    output_layer = Dense(9, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # --- 4. Callbacks ---
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        "modelo_melhor.keras", monitor="val_accuracy", save_best_only=True
    )

    # --- 5. Treinamento ---
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
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
    QNT_EPOCAS = 40
    treinar_modelo(TAM_TESTES, TAM_VALIDACAO, QNT_EPOCAS)


if __name__ == "__main__":
    main()