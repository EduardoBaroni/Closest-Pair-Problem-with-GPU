import matplotlib.pyplot as plt
import copy

y_file = open("eixoVertical", 'r')
y_data = y_file.read()

numeroAux = ""
y = []
for j in range(len(y_data)):
	if(y_data[j] != "\n"):
		numeroAux = numeroAux + y_data[j]
	else:
		numero = copy.deepcopy(numeroAux)
		y.append(float(numero))
		numeroAux = ""		

#for j in range(len(y)):
#	print((y[j]))

x_file = open("eixoHorizontal", "r")
x_data = x_file.read()

numeroAux = ""
x = []
for j in range(len(x_data)):
	if(x_data[j] != "\n"):
		numeroAux = numeroAux + x_data[j]
	else:
		numero = copy.deepcopy(numeroAux)
		x.append(int(numero))
		numeroAux = ""		

y_file = open("eixoVertical_cuda", 'r')
y_data = y_file.read()

numeroAux = ""
y_cuda = []
for j in range(len(y_data)):
	if(y_data[j] != "\n"):
		numeroAux = numeroAux + y_data[j]
	else:
		numero = copy.deepcopy(numeroAux)
		y_cuda.append(float(numero))
		numeroAux = ""		

x_file = open("eixoHorizontal_cuda", "r")
x_data = x_file.read()

numeroAux = ""
x_cuda = []
for j in range(len(x_data)):
	if(x_data[j] != "\n"):
		numeroAux = numeroAux + x_data[j]
	else:
		numero = copy.deepcopy(numeroAux)
		x_cuda.append(int(numero))
		numeroAux = ""

fig = plt.figure()
plt.plot(x,y, '-')
plt.ylabel("tempo(s)")
plt.xlabel("num_pontos")
fig.show()

fig_cuda = plt.figure()
plt.plot(x_cuda, y_cuda, '-')
plt.ylabel("tempo(s)")
plt.xlabel("num_pontos")
fig_cuda.show()

fig.savefig('graficoSeq.png')
fig_cuda.savefig('graficoCuda.png')