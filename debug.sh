#!/bin/bash

DIR_RESULTS=$1

echo "Directory created for results: $DIR_RESULTS"
mkdir $DIR_RESULTS

gerador=/gerador
seq0=/seq0
seq1=/seq1
seq2=/seq2
sed6=/sed6
paralelo0=/paralelo0
paralelo1=/paralelo1
paralelo2=/paralelo2
paralelo3=/paralelo3

if [[ ! -f "$gerador" || "$seq0" || "$seq1" || "$seq2" || "$paralelo0" || "$paralelo1" || "$paralelo2" || "$paralelo3" ]]; then
	echo "Compilando..."
	gcc gerador.c -pedantic -std=c11 -o gerador -DSCRIPT_MODE
	gcc cpp_seq_v7.c -pedantic -std=c11 -O3 -o seq0 -lm -DDEBUG
	gcc cpp_seq_v8.c -pedantic -std=c11 -O3 -o seq1 -lm -DDEBUG
	gcc cpp_seq_v9.c -pedantic -std=c11 -O3 -o seq2 -lm -DDEBUG
	gcc sed6_quick.c -pedantic -std=c11 -O3 -o sed6 -lm
	nvcc cpp_cuda_v0.cu -O3 -o paralelo0 -DDEBUG
	nvcc cpp_cuda_v1.cu -O3 -o paralelo1 -DDEBUG
	nvcc cpp_cuda_v2.cu -O3 -o paralelo2 -DDEBUG
	nvcc cpp_cuda_v3.cu -O3 -o paralelo3 -DDEBUG
	echo "Compilação finalizada"
fi

echo "Gerando cabeçalhos"

echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq0.txt
echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq1.txt
echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq2.txt

echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo0.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo1.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo2.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo3.txt	  

echo "TOTAL" > sed6.txt

echo "Informe os limites min e max em X:"
read minX
read maxX
echo "Informe os limites min e max em Y:"
read minY
read maxY
echo "Informe o número de pontos"
read num_pontos

for ((i = 0; 10 > i; i++)); do
	echo "Executando gerador"
	./gerador $num_pontos $minX $maxX $minY $maxY

	echo "Executando seq0"
	./seq0 nPontos.bin coordenadas.bin >> seq0.txt		
	
	echo "Executando seq1"	
	./seq1 nPontos.bin coordenadas.bin >> seq1.txt
	
	echo "Executando seq2"
	./seq2 nPontos.bin coordenadas.bin >> seq2.txt
	
	echo "Executando sed6"
	./sed6 nPontos.bin coordenadas.bin >> sed6.txt

	# Todas versoes paralelas sao executadas uma vez para "esquentar" o programa e entao joga para arquivo de saida
	echo "Executando paralelo0"
	
	./paralelo0 nPontos.bin coordenadas.bin
	./paralelo0 nPontos.bin coordenadas.bin >> paralelo0.txt	
	
	echo "Executando paralelo1"
	
	./paralelo1 nPontos.bin coordenadas.bin
	./paralelo1 nPontos.bin coordenadas.bin >> paralelo1.txt	

	echo "Executando paralelo2"
	
	./paralelo2 nPontos.bin coordenadas.bin
	./paralelo2 nPontos.bin coordenadas.bin >> paralelo2.txt	

	echo "Executando paralelo3"
	
	./paralelo3 nPontos.bin coordenadas.bin
	./paralelo3 nPontos.bin coordenadas.bin >> paralelo3.txt

	echo "Executando profile"

	sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo0 nPontos.bin coordenadas.bin > profile_kernels28.txt
	sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo1 nPontos.bin coordenadas.bin > profile_kernels29.txt
	sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo2 nPontos.bin coordenadas.bin > profile_kernels30.txt
	sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo3 nPontos.bin coordenadas.bin > profile_kernels31.txt

	mv *.txt $DIR_RESULTS/
	echo "Profile movido para: $DIR_RESULTS"

	/usr/bin/time -f "%M" -o Memory_File_seq7.txt ./seq0 nPontos.bin coordenadas.bin
	/usr/bin/time -f "%M" -o Memory_File_seq8.txt ./seq1 nPontos.bin coordenadas.bin
	/usr/bin/time -f "%M" -o Memory_File_seq9.txt ./seq2 nPontos.bin coordenadas.bin
	/usr/bin/time -f "%M" -o Memory_File_sed6.txt ./sed6 nPontos.bin coordenadas.bin
	/usr/bin/time -f "%M" -o Memory_File_p28.txt ./paralelo0 nPontos.bin coordenadas.bin
	/usr/bin/time -f "%M" -o Memory_File_p29.txt ./paralelo1 nPontos.bin coordenadas.bin
	/usr/bin/time -f "%M" -o Memory_File_p30.txt ./paralelo2 nPontos.bin coordenadas.bin
	/usr/bin/time -f "%M" -o Memory_File_p31.txt ./paralelo3 nPontos.bin coordenadas.bin

	mv *.txt $DIR_RESULTS/
	echo "Medidas de memória movidas para: $DIR_RESULTS"

done

echo "Execução finalizada"

mv *.txt $DIR_RESULTS/
echo "Todos resultados movidos para: $DIR_RESULTS"
