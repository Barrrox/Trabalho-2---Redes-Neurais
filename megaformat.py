"""
Esse código varre o banco de dados inteiro, executa uma operação de crop central e
redimensiona a imagem para 128x128. No fim, ele agrupa todas as imagens em
um único arquivo do numpy (.npy).

Autores: Ellen Brzozoski, João Silva, Lóra, Matheus Barros
"""

from cv2 import imread
from random import random
import numpy as np
from pathlib import Path
from time import time



def formatImage(imagemDesf):
	# definir pontos de incio e fim de um espaço de crop 1:1 aleatório
	height, width, channels = imagemDesf.shape
	startY = 0
	startX = 0
	outScale = 0

	if height > width:
		outScale = width
		startX = 0

		cropOffset = int((height - width) * 0.5)
		startY = cropOffset

	if width > height:
		outScale = height
		startY = 0

		cropOffset = int((width - height) * 0.5)
		startX = cropOffset


	# criar imagem de saída
	outRes = 128
	output = np.zeros((outRes, outRes, 3), dtype=np.uint8)


	for y in range(outRes):
		for x in range(outRes):
			output[y, x] = imagemDesf[int((y/outRes)*outScale) + startY, int((x/outRes)*outScale) + startX]
			output[y, x, 0], output[y, x, 2] = output[y, x, 2], output[y, x, 0]

	# FOI!
	return output


def getFolder(path: str) -> str:
    return Path(path).parent.name

#
def formatDataSet(caminhoParaODataSet):
	"""
	Formata o dataset inteiro de imagens, salva e retorna em forma de uma lista de matrizes de tuplas RGB.

	"""

	# Dicionario usado para transformar classe em labels
	classe_para_label = {
		"Baroque": 0,
		"Cubism": 1,
		"Expressionism": 2,
		"Impressionism": 3,
		"Minimalism": 4,
		"Post_Impressionism": 5,
		"Realism": 6,
		"Romanticism": 7,
		"Symbolism": 8
	}

	diretorio_raiz = Path(caminhoParaODataSet)

	lista_de_caminhos_imagens = []
	lista_de_caminhos_imagens.extend(diretorio_raiz.rglob("*"))

	lista_de_matrizes = []
	lista_de_classes = []
	print(f"Encontradas {len(lista_de_caminhos_imagens)} imagens.")


	# Para cada imagem na pasta de imagens
	for caminho_imagem in lista_de_caminhos_imagens:
		try:
			print(f"Processando imagem: {caminho_imagem}")

			# Abre a imagem usando o objeto Path diretamente
			imagem = imread(caminho_imagem)

			imagemFormatada = formatImage(imagem)

			# Converte para matriz NumPy e adiciona à lista
			imagemFormatada = np.array(imagemFormatada)

			lista_de_matrizes.append(imagemFormatada)
			
			# pega a label do movimento artístico para 
			classe = getFolder(str(caminho_imagem))

			lista_de_classes.append(classe_para_label[classe])

		except Exception as e:
			print(f"Erro ao processar o arquivo {caminho_imagem}: {e}")
	
	np.save('imagens_treino.npy', lista_de_matrizes)
	np.save('labels_treino.npy', lista_de_classes)

	return lista_de_matrizes, lista_de_classes

inicio = time()
formatDataSet("./BaseDeDados")
print(f"Tempo : {time() - inicio}s")

