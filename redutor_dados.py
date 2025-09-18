from time import time
import numpy as np



def redutor_dados(NUM_AMOSTRAS):

    inicio = time()

    # Carrega o conjunto de dados
    x_train = np.load("imagens_treino.npy")
    y_train = np.load("labels_treino.npy")

    x_train_reduzido = []
    y_train_reduzido = []

    # Descobrindo o indice inicial de cada movimento no array
    categorias_indice = []
    categorias_ja_visitadas = []

    for i in range(len(y_train)):
    
        if y_train[i] not in categorias_ja_visitadas:
            categorias_indice.append(i)
            categorias_ja_visitadas.append(y_train[i])
            
            # if i != 0:
            #     print(f"Categoria {y_train[i-1]}, i - 1 = {i - 1}")
            #     print(f"Categoria {y_train[i]}, i = {i}\n")
            
                

    print(f"{time() - inicio:2f}s : Indices descobertos")

    for i in range(9): # faz para as 9 categorias
        offset = categorias_indice[i]
        print(i)
        for j in range(NUM_AMOSTRAS):
            x_train_reduzido.append(x_train[j + offset])
            y_train_reduzido.append(y_train[j + offset])

    x_train_reduzido = np.array(x_train_reduzido)
    y_train_reduzido = np.array(y_train_reduzido)

    print(f"Novo database tem {len(y_train_reduzido) } imagens com {NUM_AMOSTRAS} amostras por movimento")


    np.save(f"imagens_treino_{NUM_AMOSTRAS}.npy", x_train_reduzido)
    np.save(f"labels_treino_{NUM_AMOSTRAS}.npy", y_train_reduzido)


def main():
  
    # Numero de amostras de cada movimento artístico
    NUM_AMOSTRAS = 50

    redutor_dados(NUM_AMOSTRAS)


if __name__ == "__main__":
    main()