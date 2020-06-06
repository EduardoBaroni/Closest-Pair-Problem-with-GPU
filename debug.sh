#!/bin/bash

DIR_RESULTS=$1

echo "Directory created for results: $DIR_RESULTS"
mkdir $DIR_RESULTS

echo "Compilando..."
gcc gerador.c -pedantic -std=c11 -o gerador -DSCRIPT_MODE
gcc cpp_seq_v7.c -pedantic -std=c11 -O3 -o seq_7 -lm -DDEBUG
gcc cpp_seq_v8.c -pedantic -std=c11 -O3 -o seq_8 -lm -DDEBUG
gcc cpp_seq_v9.c -pedantic -std=c11 -O3 -o seq_9 -lm -DDEBUG
gcc sed6_quick.c -pedantic -std=c11 -O3 -o sed6 -lm
#nvcc cpp_cuda_v28.cu -O3 -o paralelo28 -DDEBUG
#nvcc cpp_cuda_v29.cu -O3 -o paralelo29 -DDEBUG
#nvcc cpp_cuda_v30.cu -O3 -o paralelo30 -DDEBUG
#nvcc cpp_cuda_v31.cu -O3 -o paralelo31 -DDEBUG
#nvcc cpp_cuda_v32.cu -O3 -o paralelo32 -DDEBUG
echo "Compilação finalizada"

echo "Gerando cabeçalhos"

echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq7.txt
echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq8.txt
echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq9.txt

echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo28.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo29.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo30.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo31.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo32.txt	  

echo "TOTAL" > sed6.txt

echo "Informe os limites min e max em X:"
read minX
read maxX
echo "Informe os limites min e max em Y:"
read minY
read maxY
echo "Informe o número de pontos"
read num_pontos

for ((i = 0; 5 > i; i++)); do
	echo "Começando execução "$i

	echo "Executando gerador"
	./gerador $num_pontos $minX $maxX $minY $maxY

	echo "Executando seq7"
	./seq_7 nPontos.bin coordenadas.bin >> seq7.txt		
	
	echo "Executando seq8"	
	./seq_8 nPontos.bin coordenadas.bin >> seq8.txt
	
	echo "Executando seq9"
	./seq_9 nPontos.bin coordenadas.bin >> seq9.txt
	
	echo "Executando sed6"
	./sed6 nPontos.bin coordenadas.bin >> sed6.txt

	# Todas versoes paralelas sao executadas uma vez para "esquentar" o programa e entao joga para arquivo de saida
	#echo "Executando paralelo28"
	
	#./paralelo28 nPontos.bin coordenadas.bin
	#./paralelo28 nPontos.bin coordenadas.bin >> paralelo28.txt	
	
	#echo "Executando paralelo29"
	
	#./paralelo29 nPontos.bin coordenadas.bin
	#./paralelo29 nPontos.bin coordenadas.bin >> paralelo29.txt	

	#echo "Executando paralelo30"
	
	#./paralelo30 nPontos.bin coordenadas.bin
	#./paralelo30 nPontos.bin coordenadas.bin >> paralelo30.txt	

	#echo "Executando paralelo31"
	
	#./paralelo31 nPontos.bin coordenadas.bin
	#./paralelo31 nPontos.bin coordenadas.bin >> paralelo31.txt

	#echo "Executando paralelo32"
	
	#./paralelo32 nPontos.bin coordenadas.bin
	#./paralelo32 nPontos.bin coordenadas.bin >> paralelo32.txt
	echo "Execução " $i " finalizada"
done

echo "Execução finalizada"

#echo "Gerando médias"

#python3 gera_medias.py > medias.txt

mv *.txt $DIR_RESULTS/
echo "Todos resultados movidos para: $DIR_RESULTS"

#sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo28 nPontos.bin coordenadas.bin > profile_kernels28.txt
#sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo29 nPontos.bin coordenadas.bin > profile_kernels29.txt
#sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo30 nPontos.bin coordenadas.bin > profile_kernels30.txt
#sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo31 nPontos.bin coordenadas.bin > profile_kernels31.txt
#sudo /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -k "calculaDistancias|Forca_Bruta" -f ./paralelo32 nPontos.bin coordenadas.bin > profile_kernels32.txt

#mv *.txt $DIR_RESULTS/
#echo "All profiles results are moved to directory: $DIR_RESULTS"

#/usr/bin/time -f "%M" -o Memory_File_seq7.txt ./seq_7 nPontos.bin coordenadas.bin
#/usr/bin/time -f "%M" -o Memory_File_seq8.txt ./seq_8 nPontos.bin coordenadas.bin
#/usr/bin/time -f "%M" -o Memory_File_seq9.txt ./seq_9 nPontos.bin coordenadas.bin
#/usr/bin/time -f "%M" -o Memory_File_sed6.txt ./sed6 nPontos.bin coordenadas.bin

#mv *.txt $DIR_RESULTS/
#echo "All times CPU results are moved to directory: $DIR_RESULTS"

#/usr/bin/time -f "%M" -o Memory_File_p28.txt ./paralelo28 nPontos.bin coordenadas.bin
#/usr/bin/time -f "%M" -o Memory_File_p29.txt ./paralelo29 nPontos.bin coordenadas.bin
#/usr/bin/time -f "%M" -o Memory_File_p30.txt ./paralelo30 nPontos.bin coordenadas.bin
#/usr/bin/time -f "%M" -o Memory_File_p31.txt ./paralelo31 nPontos.bin coordenadas.bin
#/usr/bin/time -f "%M" -o Memory_File_p32.txt ./paralelo32 nPontos.bin coordenadas.bin

#mv *.txt $DIR_RESULTS/
#echo "All times GPU results are moved to directory: $DIR_RESULTS"