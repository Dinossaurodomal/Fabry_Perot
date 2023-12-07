'''
TRABALHO COMPUTACIONAL - INTRODUÇÃO À FOTÔNICA
Nome: Jessica Souza Kubit, Matheus Vieira Monteiro Nunes e Ramon Rodrigues Morello

'''

import numpy as np
import matplotlib.pyplot as plt

def calc_Fabry_Perot(r, d, wavelength, num_ondas):
    trans = 1 - r
    n = 1
    c = 299792458
    co = c / n

    t = np.linspace(0, 25, len(wavelength))
    Eo = 1
    padrao_interferencia = np.zeros(len(wavelength))

    for i in range(len(wavelength)):
        w = 2 * np.pi * co / wavelength[i]
        betha = 2 * np.pi * n / wavelength[i]
        E_vector = np.zeros((num_ondas, len(wavelength)))
        x = 1
        for j in range(num_ondas):
            E_vector[j, :] = Eo * (r**(j)) * trans * np.cos(w * t - betha * x * d)
            x = x + 2
        sum_E = np.sum(E_vector, axis=0)
        padrao_interferencia[i] = np.max(sum_E)**2

    return padrao_interferencia

# Parâmetros do interferômetro
lim_min = 1500  
lim_max = 1600  
passo = 1    

# Comprimentos de onda desejados
comp_onda1 = 1520  
comp_onda2 = 1550  

# Larguras de banda dos filtros
largura_banda1 = 10  
largura_banda2 = 1   

# Cálculo do número de passos
num_passos = int(np.floor((lim_max - lim_min) / passo) + 1)

# Inicialização das variáveis
dist_esp = 62   # Distância entre os espelhos 
Ref = 0.9       # Refletividade
n = 2

# Vetores para armazenar os resultados
intensidade1 = np.zeros(num_passos)
intensidade2 = np.zeros(num_passos)

wavelength_array = np.arange(lim_min, lim_max + passo, passo)

# Loop sobre diferentes comprimentos de onda
for i in range(num_passos):
    wavelength = wavelength_array[i]

    # Cálculo da amplitude da onda resultante para cada comprimento de onda
    amplitude = calc_Fabry_Perot(Ref, dist_esp, [wavelength], 50)[0]

    # Verificação se o comprimento de onda está dentro da faixa desejada para os filtros
    if comp_onda1 - largura_banda1 / 2 <= wavelength <= comp_onda1 + largura_banda1 / 2:
        intensidade1[i] = amplitude

    if comp_onda2 - largura_banda2 / 2 <= wavelength <= comp_onda2 + largura_banda2 / 2:
        intensidade2[i] = amplitude


# Plot
plt.figure()
plt.plot(wavelength_array, intensidade1, linewidth=1.5)
plt.plot(wavelength_array, intensidade2, linewidth=1.5)
plt.xlabel('Comprimento de Onda (nm)')
plt.ylabel('Intensidade |E|²')
plt.title('Simulação de um Interferômetro Fabry-Perot')
plt.legend(['Filtro 1', 'Filtro 2'])
plt.show()