# -*- coding: utf-8 -*-
"""
Este script realiza a otimização de hiperparâmetros para uma rede neural convolucional,
baseada na arquitetura Inception, para classificar obras de arte por movimento artístico.
A busca pelos melhores parâmetros é feita com RandomizedSearchCV sobre um subconjunto
dos dados de treino para acelerar o processo.

Autores: Ellen Brzozoski, João Silva, Lóra, Matheus Barros
"""

import numpy as np
from random import randint
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Concatenate # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from time import time

def otimizar_parametros(TAM_TESTES):
  """
  Carrega os dados, prepara um subconjunto para o treinamento e executa
  o RandomizedSearchCV para encontrar os melhores hiperparâmetros para o modelo.
  
  Args:
      TAM_TESTES (float): A proporção do conjunto de treino a ser usada 
                          para a busca de hiperparâmetros (ex: 0.15 para 15%).
  """
  inicio = time()

  # Carrega o conjunto de dados de treino completo.
  x_train = np.load("imagens_treino.npy")
  y_train = np.load("labels_treino.npy")

  print(f"Tempo = {time() - inicio:.2f}s : Dados de treino carregados")

  # Para acelerar a otimização, um subconjunto aleatório dos dados de treino é selecionado.
  # O RandomizedSearchCV será executado apenas neste subconjunto.
  x_subset = []
  y_subset = []
  for _ in range(int(len(y_train) * TAM_TESTES)):
    imagem_index = randint(0, len(y_train) - 1)
    x_subset.append(x_train[imagem_index])
    y_subset.append(y_train[imagem_index])

  print(f"Tempo = {time() - inicio:.2f}s : Subconjunto de dados criado")

  # As variáveis de treino principais agora apontam para o subconjunto de dados.
  x_train = np.array(x_subset)
  y_train = np.array(y_subset)

  # Normaliza os valores dos pixels para o intervalo [0, 1].
  x_train = x_train / 255.0

  print(f"Tempo = {time() - inicio:.2f}s : Pixels normalizados entre 0 e 1")

  # Converte os rótulos numéricos para o formato one-hot encoding (categórico).
  y_train = to_categorical(y_train, 9)

  print(f"Tempo = {time() - inicio:.2f}s : Rótulos convertidos para categorias")

  def construir_modelo_inception(optimizer='adam', dense_units=128):
    """
    Constrói e compila um modelo Keras com uma arquitetura baseada em módulos Inception.

      Args:
          optimizer (str): O otimizador a ser usado durante a compilação do modelo.
          dense_units (int): O número de neurônios na penúltima camada densa.

      Returns:
          Model: O modelo Keras compilado.
    """
    def inception_module(x, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
      # Ramo 1x1
      conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(x)
          
      # Ramo 3x3
      conv3_in = Conv2D(f2_in, (1,1), padding='same', activation='relu')(x)
      conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3_in)
          
      # Ramo 5x5
      conv5_in = Conv2D(f3_in, (1,1), padding='same', activation='relu')(x)
      conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5_in)
          
      # Ramo de Pooling
      pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
      pool_proj = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
          
      # Concatena as saídas dos quatro ramos.
      output = Concatenate(axis=-1)([conv1, conv3, conv5, pool_proj])
      return output

    input_layer = Input(shape=(128, 128, 3))
    x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(input_layer)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = Flatten()(x)
    
    # A quantidade de neurônios nesta camada é um dos hiperparâmetros a ser otimizado.
    x = Dense(dense_units, activation='relu')(x)
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
    'batch_size': [16, 32],
    'epochs': [20],

    # Parâmetros da função 'construir_modelo_inception' (prefixo 'model__')
    'model__optimizer': ['adam', 'rmsprop'],
    'model__dense_units': [128, 256, 512],
  }

  # Configura a busca aleatória com validação cruzada (cv=2).
  random_search = RandomizedSearchCV(estimator=model, param_distributions=parametros,
                                      n_iter=8, cv=2, verbose=2)

  print(f"Tempo = {time() - inicio:.2f}s : Instância do RandomizedSearch criada")

  # Inicia o processo de busca e ajuste dos hiperparâmetros.
  modelo_convergido = random_search.fit(x_train, y_train)

  print(f"Tempo = {time() - inicio:.2f}s : Otimização de hiperparâmetros concluída")

  # Exibe os melhores resultados encontrados.
  print(f"\nMelhores parâmetros encontrados: {modelo_convergido.best_params_}")
  print(f"Melhor score de validação cruzada: {modelo_convergido.best_score_:.4f}")


def main():
  """
  Ponto de entrada do script. Define a proporção do dataset a ser usada
  e chama a função de otimização.
  """
  # Define que 15% do conjunto de treino será usado para a busca de hiperparâmetros.
  TAM_TESTES = 0.15
  otimizar_parametros(TAM_TESTES)

# Bloco padrão para garantir que main() seja executado apenas quando o script for chamado diretamente.
if __name__ == "__main__":
  main()