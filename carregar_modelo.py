from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.widgets import Button 
import numpy as np
import random

def carregar_modelo(QNT_TESTES):
    """
    Carrega um modelo Keras treinado, avalia sua performance com uma matriz de confusão
    e inicia uma janela interativa para testar predições em imagens aleatórias.

    Args:
        QNT_TESTES (int): O número de amostras a serem selecionadas aleatoriamente
                          do conjunto de treino para compor o conjunto de teste.
    """
    # Carrega o modelo pré-treinado a partir do arquivo 'model.keras'.
    model = load_model('model.keras')

    # --- Etapa 1: Geração da Matriz de Confusão ---
    
    # Carrega as imagens e os rótulos do conjunto de dados de treino.
    # Usaremos estes dados como base para criar um conjunto de teste aleatório.
    x_train = np.load("imagens_treino.npy")
    y_train = np.load("labels_treino.npy")

    # Cria um conjunto de teste selecionando amostras aleatórias do conjunto de treino.
    # Isso é útil para uma validação rápida sem um arquivo de teste dedicado.
    x_test = []
    y_test = []
    for _ in range(QNT_TESTES):
        imagem_index = random.randint(0, len(y_train) - 1)
        x_test.append(x_train[imagem_index])
        y_test.append(y_train[imagem_index])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Normaliza os valores dos pixels para o intervalo [0, 1], que é o formato esperado pela rede.
    x_test = x_test / 255.0

    # Realiza as predições no conjunto de teste e pega o índice da classe com maior probabilidade.
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Gera a matriz de confusão para comparar os resultados previstos com os reais.
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plota a matriz de confusão para visualização da performance do modelo.
    ConfusionMatrixDisplay(conf_matrix).plot(cmap='Blues')
    plt.title("Matriz de Confusão")
    print("Matriz de confusão exibida com sucesso.")

    # --- Etapa 2: Interface Interativa de Predição ---
    
    # Cria uma nova figura com dois subplots: um para a imagem, outro para o gráfico de probabilidades.
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8, 4))
    
    def mostrar_proxima_imagem(event):
        """
        Callback para o botão 'Próxima Imagem'. Seleciona uma nova imagem,
        faz a predição e atualiza os subplots na tela.
        
        O parâmetro 'event' é exigido pelo Matplotlib, por um motimo que apenas Alan Turing sabia.
        """
        # Limpa os eixos para exibir a nova imagem e o novo gráfico.
        ax_img.clear()
        ax_bar.clear()
        
        # Seleciona um índice aleatório do nosso conjunto de teste para a predição.
        random_idx = np.random.randint(0, x_test.shape[0])
        # Adiciona uma dimensão de 'batch' (lote) à imagem, pois o modelo espera o formato (1, 255, 255, 3).
        new_input = x_test[random_idx].reshape(1, 255, 255, 3)

        # Executa a predição na imagem selecionada.
        prediction = model.predict(new_input)
        predicted_class = np.argmax(prediction)

        # Pega a classe verdadeira (original) da imagem para comparação.
        original_class = y_test[random_idx]

        # Mostra a imagem de teste no subplot da esquerda.
        ax_img.imshow(new_input[0], cmap='gray')
        ax_img.set_title(f'Classe Original: {original_class}')
        ax_img.axis('off') # Remove os eixos para uma visualização mais limpa.

        # Plota as probabilidades no subplot da direita.
        ax_bar.bar([0, 1], [prediction[0, predicted_class], 1 - prediction[0, predicted_class]], color=['blue', 'red'])
        ax_bar.set_xticks([0, 1], ['Classe Predita', 'Outra Classe'])
        ax_bar.set_title(f'Classe Predita: {predicted_class}')
        
        # Redesenha o canvas da figura para aplicar as atualizações.
        fig.canvas.draw_idle()

        # Imprime as informações no console.
        print("-" * 30)
        print(f"Classe prevista: {predicted_class}")
        print(f"Classe original: {original_class}")

    # Ajusta o posicionamento dos subplots para dar espaço ao botão.
    fig.subplots_adjust(bottom=0.2)
    
    # Define a área onde o botão será criado. Formato: [esquerda, baixo, largura, altura].
    ax_botao = fig.add_axes([0.4, 0.05, 0.2, 0.075])
    
    # Cria o widget do botão.
    botao_proxima = Button(ax_botao, 'Próxima Imagem')
    
    # Associa a função 'mostrar_proxima_imagem' ao evento de clique do botão.
    botao_proxima.on_clicked(mostrar_proxima_imagem)
    
    # Chama a função uma vez no início para já exibir a primeira imagem sem precisar clicar.
    mostrar_proxima_imagem(None)
    
    # Exibe todas as janelas geradas (matriz e interface interativa).
    plt.show()

def main():
    """
    Ponto de entrada principal do script.
    """
    # Define a quantidade de imagens a serem usadas para gerar a matriz de confusão.
    QNT_TESTES = 18000
    carregar_modelo(QNT_TESTES)

# Garante que o script só será executado quando chamado diretamente.
if __name__ == "__main__":
    main()