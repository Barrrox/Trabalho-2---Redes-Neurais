from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.widgets import Button 
import numpy as np
import random


def carregar_modelo(QNT_TESTES):
    ##Parte 2: Carregando o Modelo e Fazendo Previsões

    #1. Carregando o Modelo Salvo

    # Carrega o modelo salvo
    model = load_model('model.keras')

    #2. Matriz de confusão do conjunto de treino

    # Carrega o conjunto de dados para teste
    x_train = np.load("imagens_treino.npy")
    y_train = np.load("labels_treino.npy")

    # Criar Xtest e Ytest aleatoriamente
    x_test = []
    y_test = []
    for _ in range(QNT_TESTES):
        imagem_index = random.randint(0, len(y_train) - 1)
        x_test.append(x_train[imagem_index])
        y_test.append(y_train[imagem_index])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Normaliza os dados de teste
    x_test = x_test / 255.0

    # Faz previsões no conjunto de teste
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Calcula a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Exibe a matriz de confusão
    ConfusionMatrixDisplay(conf_matrix).plot(cmap='Blues')
    plt.title("Matriz de Confusão")
    print("Matriz de confusão exibida com sucesso.")

    # 3. Fazendo Previsões com o Modelo Carregado de forma interativa

    # Cria a figura e os eixos que serão atualizados
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Função que será chamada pelo botão
    def mostrar_proxima_imagem():
        # Limpa os eixos antes de desenhar a nova imagem/gráfico
        ax_img.clear()
        ax_bar.clear()
        
        # Simula uma nova entrada: seleciona uma imagem aleatória do conjunto de teste
        random_idx = np.random.randint(0, x_test.shape[0])
        new_input = x_test[random_idx].reshape(1, 255, 255, 3)  # Redimensiona para o formato esperado pelo modelo

        # Faz a previsão
        prediction = model.predict(new_input)
        predicted_class = np.argmax(prediction)

        # Obtém a classe original
        original_class = y_test[random_idx]

        # Exibe a imagem de entrada e as classes nos eixos corretos
        ax_img.imshow(new_input[0], cmap='gray')
        ax_img.set_title(f'Classe Original: {original_class}')
        ax_img.axis('off')

        ax_bar.bar([0, 1], [prediction[0, original_class], 1 - prediction[0, original_class]], color=['blue', 'red'])
        ax_bar.set_xticks([0, 1], ['Classe Predita', 'Probabilidade'])
        ax_bar.set_title(f'Classe Predita: {predicted_class}')
        
        # Redesenha a figura para mostrar as atualizações
        fig.canvas.draw_idle()

        print("-" * 30)
        print(f"Classe prevista: {predicted_class}")
        print(f"Classe original: {original_class}")

    # Ajusta o layout para criar espaço para o botão na parte inferior
    fig.subplots_adjust(bottom=0.2)
    
    # Define a posição e o tamanho do botão [esquerda, baixo, largura, altura]
    ax_botao = fig.add_axes([0.4, 0.05, 0.2, 0.075])
    
    # Cria o botão e define seu rótulo
    botao_proxima = Button(ax_botao, 'Próxima Imagem')
    
    # Conecta o evento de clique do botão à nossa função
    botao_proxima.on_clicked(mostrar_proxima_imagem)
    
    # Chama a função uma vez manualmente para exibir a primeira imagem
    mostrar_proxima_imagem()
    
    # Exibe todas as figuras (Matriz de Confusão e a interface interativa)
    plt.show()

    # --- FIM DAS ALTERAÇÕES ---

def main():

    # Quantidade de imagens de teste para criar a matriz de confusão
    QNT_TESTES = 1000

    carregar_modelo(QNT_TESTES)


if __name__ == "__main__":
    main()