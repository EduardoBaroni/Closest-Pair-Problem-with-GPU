#!/bin/bash

DIR_RESULTS=$1

echo "Diretorio criado para os resultados: $DIR_RESULTS"
mkdir $DIR_RESULTS

echo "Compilando..."
gcc gerador.c -pedantic -std=c11 -o gerador -DSCRIPT_MODE -DtresD
gcc cpp_seq_3D.c -pedantic -std=c11 -O3 -o seq_3D -lm -DDEBUG
nvcc cpp_cuda_3D.cu -O3 -o paralelo_3D -DDEBUG
echo "Compilação finalizada"

echo "Gerando cabeçalhos"

echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq_3D.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo_3D.txt  

echo "Informe os limites min e max em X:"
read minX
read maxX
echo "Informe os limites min e max em Y:"
read minY
read maxY
echo "Informe os limites min e max em Z:"
read minZ
read maxZ
echo "Informe o número de pontos"
read num_pontos

for ((i = 0; 5 > i; i++)); do
	echo "Executando gerador"
	./gerador $num_pontos $minX $maxX $minY $maxY $minZ $maxZ

	echo "Executando seq_3D"
	./seq_3D nPontos.bin coordenadas.bin >> seq_3D.txt			
	
	echo "Executando paralelo_3D"
	# Executa uma vez para esquentar o programa e entao joga para arquivo de saida
	./paralelo_3D nPontos.bin coordenadas.bin	
	./paralelo_3D nPontos.bin coordenadas.bin >> paralelo_3D.txt
	
	echo "Execução finalizada"
done

echo "Gerando médias"

python3 gera_medias_3D.py > medias.txt

echo "Movendo todos resultados para o diretorio: $DIR_RESULTS"
mv *.txt $DIR_RESULTS/
