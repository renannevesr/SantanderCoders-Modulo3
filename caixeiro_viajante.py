#%%
import pandas as pd
import math

def ajusta_coordenada(coordenada):
    # Troca vírgula por ponto
    coordenada = coordenada.replace(',','.')

    # Converte para float
    coordenada = float(coordenada)

    # Ajusta coordenada para ficar entre -90 e 90
    if coordenada <= -100:
        pot = math.floor(math.log10(abs(coordenada)))
        coordenada = coordenada / 10**(pot-1)
    return float(coordenada)

df = pd.read_csv('data/escolas.csv', usecols=['lat','lon'])
df['lat'] = df['lat'].apply(ajusta_coordenada)
df['lon'] = df['lon'].apply(ajusta_coordenada)

#%% plot coords
import matplotlib.pyplot as plt
plt.scatter(df['lon'], df['lat'])
plt.show()


#%% Caixeiro Viajante
import numpy as np

def distancia_euclidiana(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

def calcula_distancias(coordenadas):
    
    distancias = np.zeros((len(coordenadas), len(coordenadas)))
    for i in range(len(coordenadas)):
        for j in range(len(coordenadas)):
            distancias[i,j] = distancia_euclidiana(coordenadas[i], coordenadas[j])

    sequencia = np.arange(len(coordenadas))
    matrix_combinacoes = np.array([sequencia]*len(coordenadas))

    for i in range(len(coordenadas)):
        matrix_combinacoes[i] = matrix_combinacoes[i][np.argsort(distancias[i])]
    
    return matrix_combinacoes

def vizinho_mais_proximo(escola, visitados, proximos):
    visitados[escola] = 1
    for i in range(len(visitados)):
        proxima_escola = proximos[escola][i]
        # print(escola, proxima_escola, visitados[proxima_escola])
        if visitados[proxima_escola] == 0:
            return [escola] + vizinho_mais_proximo(proxima_escola, visitados, proximos)
    return [escola]

def caixeiro_viajante(coordenadas):
    proximos = calcula_distancias(coordenadas)
    
    visitados = np.zeros(len(coordenadas))
    
    first = np.random.randint(0, len(coordenadas))
    caminho = vizinho_mais_proximo(first, visitados, proximos)

    caminho.append(caminho[0])
    return caminho

coordenadas = df[['lat','lon']].to_numpy()

caminho = caixeiro_viajante(coordenadas)
caminho
# %%
import matplotlib.pyplot as plt
plt.scatter(df['lon'], df['lat'])
plt.plot(df['lon'][caminho], df['lat'][caminho], 'r')
plt.savefig('caminho.png')
plt.show()
# %%
