#!/bin/bash

DIR_RESULTS=$1

echo "Directory created for results: $DIR_RESULTS"
mkdir $DIR_RESULTS

echo "Compilando..."
gcc gerador_3D.c -pedantic -std=c11 -o gerador
gcc cpp_seq_3d.c -pedantic -std=c11 -O3 -o seq_3D -lm -DDEBUG -DCONTADOR
nvcc cpp_cuda_3D.cu -O3 -o paralelo3D -DDEBUG

echo "Compilação finalizada"

echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq3D.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo3D.txt

echo "Executando..."
echo "Executando seq3D"
for (( i = 0; 5 > i; i++)); do
	./seq_3D nPontos.bin coordenadas.bin >> seq3D.txt			
	echo "#"
done
echo "Executando paralelo3D"
for (( i = 0; 6 > i; i++)); do
	./paralelo3D nPontos.bin coordenadas.bin >> paralelo3D.txt	
	echo "#"		
done

echo "Execução finalizada"

mv *.txt $DIR_RESULTS/
echo "All files results are moved to directory: $DIR_RESULTS"

sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo3D nPontos.bin coordenadas.bin > profile_kernels3D.txt

mv *.txt $DIR_RESULTS/
echo "All profiles results are moved to directory: $DIR_RESULTS"

/usr/bin/time -f "%M" -o Memory_File_seq3D.txt ./seq_3D nPontos.bin coordenadas.bin


mv *.txt $DIR_RESULTS/
echo "All times CPU results are moved to directory: $DIR_RESULTS"

/usr/bin/time -f "%M" -o Memory_File_p28.txt ./paralelo3D nPontos.bin coordenadas.bin

mv *.txt $DIR_RESULTS/
echo "All times GPU results are moved to directory: $DIR_RESULTS"
