"""
# Classificação de obras de arte por movimento artístico

Autores: Ellen Brzozoski, João Silva, Lóra, Matheus Barros
"""

from numpy import load, array 
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Concatenate, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from time import time


def treinar_modelo(TAM_TESTES, QNT_EPOCAS):
 
  inicio = time()

  # Carrega o conjunto de dados completo a partir dos arquivos .npy
  imagens = load("imagens_treino.npy")
  labels = load("labels_treino.npy")

  print(f"Tempo = {time() - inicio:2f}s : Dados carregados")

  # Divide os dados em conjuntos de treino e teste de forma que
  # um não tenha imagens do outro
  x_train, x_test, y_train, y_test = train_test_split(
      imagens, 
      labels, 
      test_size=TAM_TESTES, 
      # Garante que a divisão seja sempre a mesma 
      random_state=42 
  )
  
  print(f"Tempo = {time() - inicio:2f}s : Dados de treino e teste separados corretamente")

  # Converte as listas de imagens e rótulos para arrays NumPy
  x_test = array(x_test)
  y_test = array(y_test)

  print(f"Tempo = {time() - inicio:2f}s : Conversão concluida")

  # Normaliza os valores dos pixels para o intervalo [0, 1]
  # serve para otimizar o treino
  x_train = x_train / 255.0
  x_test = x_test / 255.0

  print(f"Tempo = {time() - inicio:2f}s : Pixels normalizados entre 0 e 1")

  # Converte os rótulos numéricos (labels) para o formato one-hot encoding
  y_train = to_categorical(y_train, 9)
  y_test = to_categorical(y_test, 9)

  print(f"Tempo = {time() - inicio:2f}s : Rótulos covertidos para categorias")

  def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    """
    Define um bloco de construção "Inception", que aplica filtros de diferentes 
    tamanhos em paralelo e concatena os resultados.

    Argumentos:
      x: Tensor de entrada.
      filters_...: Número de filtros para cada camada convolucional dentro do módulo.
    """
    # Ramo 1: Convolução simples com filtro 1x1
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    # Ramo 2: filtro 1x1 -> convolução 3x3
    # Isso reduz a profundidade do tensor, o que melhora a eficiência pois reduz o número de operações 
    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)

    # Ramo 3: filtro 1x1 -> convolução 5x5
    conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)

    # Ramo 4: MaxPooling -> convolução 1x1
    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj_conv = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    # Concatena os mapas de características de todos os ramos
    output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool_proj_conv])

    return output

  # Define a camada de entrada com o formato das imagens (255x255 pixels, 3 canais de cor)
  input_layer = Input(shape=(255, 255, 3))

  # Bloco convolucional inicial para extração de características e redução da dimensionalidade
  x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(input_layer)
  x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

  # Adiciona o primeiro módulo Inception com seus respectivos números de filtro
  x = inception_module(x,
                       filters_1x1=64,
                       filters_3x3_reduce=96,
                       filters_3x3=128,
                       filters_5x5_reduce=16,
                       filters_5x5=32,
                       filters_pool_proj=32)

  # Adiciona um segundo módulo Inception para aumentar a profundidade e a capacidade do modelo
  x = inception_module(x,
                       filters_1x1=128,
                       filters_3x3_reduce=128,
                       filters_3x3=192,
                       filters_5x5_reduce=32,
                       filters_5x5=96,
                       filters_pool_proj=64)

  x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

  # Prepara o tensor para a classificação e define as camadas densas (fully-connected)
  x = Flatten()(x)
  x = Dense(128, activation='relu')(x)
  
  # A taxa de 0.5 significa que 50% dos neurônios da camada anterior serão "desligados"
  # aleatoriamente em cada passo do treinamento.
  x = Dropout(0.5)(x)
  
  output_layer = Dense(9, activation='softmax')(x)

  # Cria o modelo final conectando a camada de entrada com a de saída
  model = Model(inputs=input_layer, outputs=output_layer, name='inception_style_model')

  # Imprime um resumo da arquitetura do modelo no console
  model.summary()

  # Configura o otimizador, a função de perda e as métricas para o treinamento
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  print(f"Tempo = {time() - inicio:2f}s : Modelo compilado")

  # Treina o modelo
  model.fit(x_train, y_train, epochs=QNT_EPOCAS, validation_data=(x_test, y_test))
  # O 'validation_data' é usado ao final de cada época para avaliar o desempenho
  # do modelo em dados não vistos, ajudando a monitorar o overfitting.

  print(f"Tempo = {time() - inicio:2f}s : Modelo treinado")

  model.save('model.keras')

  print(f"Tempo = {time() - inicio:2f}s : Modelo salvo")


def main():
 
  # Define a proporção do dataset a ser usada como conjunto de teste (ex: 0.1 para 10%)
  TAM_TESTES = 1/10

  # Define o número de épocas (ciclos completos de treinamento sobre o dataset)
  QNT_EPOCAS = 30

  treinar_modelo(TAM_TESTES, QNT_EPOCAS)


if __name__ == "__main__":
  main()