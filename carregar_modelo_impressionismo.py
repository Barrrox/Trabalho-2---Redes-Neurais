from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import random


def carregar_modelo(QNT_TESTES):
	##Parte 2: Carregando o Modelo e Fazendo Previsões

	#1. Carregando o Modelo Salvo

	# Carrega o modelo salvo
	model = load_model('model.keras')

	#2. Matriz de confusão do conjunto de treino

	# Carrega o conjunto de dados MNIST para teste
	x_train = np.load("imagens_treino_impressionismo.npy")
	y_train = np.load("labels_treino_impressionismo.npy")

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
	print("Matriz de confusão exibida com sucesso.")

	#3. Fazendo Previsões com o Modelo Carregado

	# Simula uma nova entrada: seleciona uma imagem aleatória do conjunto de teste
	random_idx = np.random.randint(0, x_test.shape[0])
	new_input = x_test[random_idx].reshape(1, 255, 255, 3)  # Redimensiona para o formato esperado pelo modelo

	# Faz a previsão
	prediction = model.predict(new_input)
	predicted_class = np.argmax(prediction)

	# Obtém a classe original
	original_class = y_test[random_idx]

	# Exibe a imagem de entrada e as classes
	plt.figure(figsize=(6, 3))
	plt.subplot(1, 2, 1)
	plt.imshow(new_input[0], cmap='gray')
	plt.title(f'Classe Original: {original_class}')
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.bar([0, 1], [prediction[0, original_class], 1 - prediction[0, original_class]], color=['blue', 'red'])
	plt.xticks([0, 1], ['Classe Predita', 'Probabilidade'])
	plt.title(f'Classe Predita: {predicted_class}')

	plt.show()

	print(f"Classe prevista: {predicted_class}")
	print(f"Classe original: {original_class}")


def main():

	# Quantidade de imagens de teste para criar a matriz de confusão
	QNT_TESTES = 1000

	carregar_modelo(QNT_TESTES)


if __name__ == "__main__":
	main()