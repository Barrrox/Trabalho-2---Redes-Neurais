"""
Esse código carrega o modelo e o conjunto de treino para gerar uma
matriz de confusão e fazer a previsão de 9 imagens, uma de cada
movimento artístico

Autores: Ellen Brzozoski, João Silva, Lóra, Matheus Barros
"""

from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split # Importar a função de divisão
import matplotlib.pyplot as plt
from matplotlib.widgets import Button # Importar o widget de botão
import numpy as np
import random


label_para_classe = {
    0: 'Baroque', 
    1: 'Cubism', 
    2: 'Expressionism', 
    3: 'Impressionism', 
    4: 'Minimalism', 
    5: 'Post_Impressionism', 
    6: 'Realism', 
    7: 'Romanticism', 
    8: 'Symbolism'
    }


def carregar_modelo(TAM_TESTES):
    """
    Carrega um modelo Keras treinado, avalia sua performance com uma matriz de confusão
    e exibe uma janela interativa com a predição de 9 imagens, uma de cada categoria.

    Args:
        TAM_TESTES (float): A proporção do conjunto de dados a ser usada como
                            conjunto de teste para a matriz de confusão.
    """
    # Carrega o modelo pré-treinado a partir do arquivo 'model.keras'.
    model = load_model('modelo.keras')

    # --- Etapa 1: Geração da Matriz de Confusão ---
    
    # Carrega as imagens e os rótulos do conjunto de dados completo.
    imagens = np.load("imagens_treino.npy")
    labels = np.load("labels_treino.npy")

    # Separa os dados em um conjunto de treino (para a etapa 2) e um de teste (para a matriz).
    # Este passo é idêntico ao do script de treinamento para garantir consistência.
    x_train, x_test, y_train, y_test = train_test_split(
        imagens, labels, test_size=TAM_TESTES, random_state=42, stratify=labels
    )
    
    # Normaliza os valores dos pixels do conjunto de teste para o intervalo [0, 1].
    x_test_normalized = x_test / 255.0

    # Realiza as predições no conjunto de teste.
    y_pred = np.argmax(model.predict(x_test_normalized), axis=1)

    # Gera a matriz de confusão.
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Pega os nomes das classes do dicionário para usar como rótulos.
    nomes_das_classes = list(label_para_classe.values())

    # 1. Cria uma figura e eixos com tamanho customizado para a matriz de confusão.
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))

    # 2. Cria o objeto de exibição da matriz com os nomes das classes.
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=nomes_das_classes)

    # 3. Plota a matriz nos eixos que criamos (ax=ax_cm).
    disp.plot(cmap='Blues', xticks_rotation=40, ax=ax_cm)

    # 3.1 
    plt.setp(ax_cm.get_xticklabels(), ha="right", rotation_mode="anchor")

    # 4. Define o título nos eixos específicos da matriz
    ax_cm.set_title("Matriz de Confusão")

    # 5. Ajusta o layout para garantir que tudo (incluindo os rótulos) caiba na figura
    fig_cm.tight_layout()

    print("Matriz de confusão exibida com sucesso.")

    # --- Etapa 2: Janela de Predição Interativa com 9 Imagens ---

    # Encontra as classes únicas no conjunto de treino.
    classes_to_show = np.unique(y_train)
        
    # Cria uma nova figura para exibir as 9 imagens em uma grade 3x3.
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle('Predição de 9 Imagens (Uma de Cada Categoria)', fontsize=16)

    # Achata o array de eixos para facilitar a iteração.
    axes = axes.flatten()

    def atualizar_grid(event):
        """
        Função chamada pelo botão. Limpa o grid e preenche com 9 novas
        imagens aleatórias e suas predições.
        """
        for i, class_label in enumerate(classes_to_show):
            ax = axes[i]
            ax.clear() # Limpa o subplot antes de desenhar a nova imagem

            # Encontra TODOS os índices de imagens que correspondem à classe atual.
            indices_da_classe = np.where(y_train == class_label)[0]
            # Escolhe um índice ALEATÓRIO dentro dessa lista.
            idx = random.choice(indices_da_classe)
            
            image = x_train[idx]
            real_class = y_train[idx]

            # Prepara a imagem para o modelo (normaliza e adiciona dimensão de batch).
            image_normalized = image / 255.0
            image_for_prediction = np.expand_dims(image_normalized, axis=0)

            # Faz a predição.
            prediction = model.predict(image_for_prediction)
            predicted_class = np.argmax(prediction)

            # Exibe a imagem no subplot correspondente.
            ax.imshow(image)
            ax.set_title(f"Real: {label_para_classe[real_class]}\nPredita: {label_para_classe[predicted_class]}")
            ax.axis('off') # Remove os eixos para uma visualização limpa.
        
        # Redesenha a figura para que as alterações apareçam.
        fig.canvas.draw_idle()

    # Ajusta o layout para criar espaço para o botão e entre as imagens.
    fig.subplots_adjust(bottom=0.2, hspace=0.4)

    # Define a área onde o botão será criado. Formato: [esquerda, baixo, largura, altura].
    ax_botao = fig.add_axes([0.4, 0.05, 0.2, 0.075])

    # Cria o widget do botão e atribui um nome a ele.
    botao_proximo = Button(ax_botao, 'Próximas Imagens')

    # Associa a função 'atualizar_grid' ao evento de clique do botão.
    botao_proximo.on_clicked(atualizar_grid)
    
    # Chama a função uma vez no início para já exibir o primeiro grid de imagens.
    atualizar_grid(None)
    
    # Exibe todas as janelas geradas (matriz e a grade de imagens).
    plt.show()

def main():
    """
    Ponto de entrada principal do script.
    """
    # Define a proporção de imagens a serem usadas para gerar a matriz de confusão.
    TAM_TESTES = 0.10
    carregar_modelo(TAM_TESTES)

# Garante que o script só será executado quando chamado diretamente.
if __name__ == "__main__":
    main()