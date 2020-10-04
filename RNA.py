# importação das bibliotecas necessárias

# pybrain
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


# gráficos 
import matplotlib.pyplot as plt
import numpy as np

# função para carregar os dados de treinamento
def getData( path ):
    #Open file
    file = open( path, "r" )
    
    data = []    
    
    for linha in file:        # obtem cada linha do arquivo
      linha = linha.rstrip()  # remove caracteres de controle, \n
      digitos = linha.split(" ")  # pega os dígitos
      for numero in digitos:   # para cada número da linha
        data.append( numero )  # add ao vetor de dados  
    
    file.close()
    return data


# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( 45, 1000, 1000, 1 )    # define network 
dataSet = SupervisedDataSet( 45, 1 )  # define dataSet

'''
arquivos = ['1.txt', '1a.txt', '1b.txt', '1c.txt',
            '1d.txt', '1e.txt', '1f.txt']
'''  
arquivos = [
    '1.txt', '1a.txt', '1b.txt', '1c.txt', '1d.txt', '1e.txt', '1f.txt',
    '2.txt', '2a.txt', '2b.txt', '2c.txt', '2d.txt', '2e.txt', '2f.txt',
    '3.txt', '3a.txt', '3b.txt', '3c.txt', '3d.txt', '3e.txt', '3f.txt',
    '4.txt', '4a.txt', '4b.txt', '4c.txt', '4d.txt', '4e.txt', '4f.txt',
    '5.txt', '5a.txt', '5b.txt', '5c.txt', '5d.txt', '5e.txt', '5f.txt',
    '6.txt', '6a.txt', '6b.txt', '6c.txt', '6d.txt', '6e.txt', '6f.txt',
    '7.txt', '7a.txt', '7b.txt', '7c.txt', '7d.txt', '7e.txt', '7f.txt',
    '8.txt', '8a.txt', '8b.txt', '8c.txt', '8d.txt', '8e.txt', '8f.txt',
    '9.txt', '9a.txt', '9b.txt', '9c.txt', '9d.txt', '9e.txt', '9f.txt',
    '0.txt', '0a.txt', '0b.txt', '0c.txt', '0d.txt', '0e.txt', '0f.txt'
]          
# a resposta do número
resposta = [ 
    [1], [1], [1], [1], [1], [1], [1],
    [2], [2], [2], [2], [2], [2], [2],
    [3], [3], [3], [3], [3], [3], [3],
    [4], [4], [4], [4], [4], [4], [4],
    [5], [5], [5], [5], [5], [5], [5],
    [6], [6], [6], [6], [6], [6], [6],
    [7], [7], [7], [7], [7], [7], [7],
    [8], [8], [8], [8], [8], [8], [8],
    [9], [9], [9], [9], [9], [9], [9],
    [0], [0], [0], [0], [0], [0], [0]

] 
#resposta = [[1], [1], [1], [1], [1], [1], [1]] 

i = 0
for arquivo in arquivos:           # para cada arquivo de treinamento
    data =  getData( arquivo )            # pegue os dados do arquivo
    dataSet.addSample( data, resposta[i] )  # add dados no dataSet
    i = i + 1


# trainer
trainer = BackpropTrainer( network, dataSet )
error = 1
iteration = 0
outputs = []
file = open("outputs.txt", "w") # arquivo para guardar os resultados

while error > 0.001: # 10 ^ -3
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    file.write( str(error)+"\n" )

file.close()

# Fase de teste
arquivos = ['1- test.txt']
for arquivo in arquivos:
    data =  getData( arquivo )
    print ( network.activate( data ) )


# plot graph
plt.ioff()
plt.plot( outputs )
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()

