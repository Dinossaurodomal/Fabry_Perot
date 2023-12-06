import numpy as np
import math
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------#
#definindo parâmetros gerais
d = 20 #distância entre os espelhos
R = 0.9 #Refletividade do segundo espelho
T = 1-R
n = 1   #indice de refração do segundo espelho
c = 299792458
c0 = c / n
lamb = np.linspace(1500e-9, 1600e-9, 1000)
beta = 2*np.pi*n
w = 2*np.pi*c0
tempo = np.linspace(0, 25e-15, len(lamb))
E0 = 1
#--------------------------------------------------------------------------------------------------------------------#
#definindo largura de banda e limites da banda
lamb_min = 1500
lamb_max = 1600
lamb_step = 1
lamb1 = 1520
lamb2 = 1550
band_width1 = 10
band_width2 = 1
num_steps = np.floor((lamb_max - lamb_min) / lamb_step) + 1
Num_steps = int(num_steps)

#--------------------------------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------------------------------#
intensidade1 = np.zeros(1, num_steps)
intensidade2 = np.zeros(1, num_steps)

# Loop sobre diferentes comprimentos de onda
'''
for i in range(1, len(lamb)):
	w = 2 * np.pi * c0 / lamb[i]
	betha = 2 * np.pi * n / lamb[i]
	E_vector = np.zeros((Num_steps - 1), len(lamb))
	x = 1; 
	for j in range(1, Num_steps):
		E_vector[j, :] = E0*T*np.power(R,(j-1))*np.cos((w*tempo)-(beta*x*d))
		x = x+1   
	amplitude = np.sum(E_vector)
	
	if abs(lamb - lamb1) <= band_width1 / 2:
		intensidade1[i] = np.power(amplitude,2)
     
 '''

    
       




for i in range(1,Num_steps):
    wave_length = lamb_min + (i - 1) * lamb_step
    amplitude = np.zeros((Num_steps - 1))
    amplitude[i] = E0*T*np.power(R,(i-1))*np.cos((w*tempo)-(beta*(2*i-1)*d))
    Amplitude = np.sum(amplitude)

    
    if abs(lamb - lamb1) <= band_width1 / 2:
        intensidade1[i] = Amplitude^2
    

    if abs(lamb - lamb2) <= band_width2 / 2:
        intensidade2[i] = Amplitude^2

        

lamb_range = np.arange(lamb_min, lamb_max + lamb_step, lamb_step)
intensity1 = np.random.rand(len(lamb_range))
intensity2 = np.random.rand(len(lamb_range))

# Plotagem dos resultados
plt.figure()
plt.plot(lamb_range, np.abs(intensity1), 'b', linewidth=2)
plt.plot(lamb_range, np.abs(intensity2), 'r', linewidth=2)
plt.xlabel('Comprimento de Onda (nm)')
plt.ylabel('Intensidade')
plt.title('Fabry-Perot')
plt.legend(['Filtro 1', 'Filtro 2'])
plt.show()


