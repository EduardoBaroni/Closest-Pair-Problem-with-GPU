def gera_medias_paralelo_debug(arquivo):
	file = open(arquivo).readlines()
	file.pop(0) # excluindo a primeira linha

	valores = [0,0,0,0,0,0,0,0]

	#for line in file:
	#	print(line)
	
	i = 0
	for line in file:
		if( i != 0):
			valores[0] += float(line[0:7])/5
			valores[1] += float(line[13:20])/5
			valores[2] += float(line[30:37])/5
			valores[3] += float(line[49:56])/5
			valores[4] += float(line[64:72])/5
			valores[5] += float(line[79:87])/5
			valores[6] += float(line[93:101])/5
			valores[7] += float(line[104:112])/5
		i+=1

	print(arquivo, end = ': ')
	for e in valores:
		print('%.5f seg'%e, end = ':')
	print('\n')

def gera_medias_sequencial_debug(arquivo):
	file = open(arquivo).readlines()
	file.pop(0) # excluindo a primeira linha

	valores = [0,0,0,0,0,0]

	#for line in file:
	#	print(line)
	
	for line in file:
			valores[0] += float(line[0:10])/5
			valores[1] += float(line[17:25])/5
			valores[2] += float(line[28:39])/5
			valores[3] += float(line[50:60])/5
			valores[4] += float(line[68:80])/5
			valores[5] += float(line[82:95])/5

	print(arquivo, end = ': ')
	for e in valores:
		print('%.5f'%e, end = ':')
	print('\n')
	
if __name__ == '__main__':
	gera_medias_paralelo_debug("paralelo_28.txt")
	gera_medias_paralelo_debug("paralelo_29.txt")
	gera_medias_paralelo_debug("paralelo_30.txt")
	gera_medias_paralelo_debug("paralelo_31.txt")

	gera_medias_sequencial_debug("seq_7.txt")
	gera_medias_sequencial_debug("seq_8.txt")
	gera_medias_sequencial_debug("seq_9.txt")
