#!/bin/bash

echo "Compilando..."
gcc gerador_bin_v5.c -pedantic -std=c11 -o gerador
gcc cpp_seq_v7.c -pedantic -std=c11 -O3 -o seq_7 -lm -DDEBUG
gcc cpp_seq_v8.c -pedantic -std=c11 -O3 -o seq_8 -lm -DDEBUG
gcc cpp_seq_v9.c -pedantic -std=c11 -O3 -o seq_9 -lm -DDEBUG
gcc sed6_quick.c -pedantic -std=c11 -O3 -o sed6 -lm
nvcc cpp_cuda_v28.cu -O3 -o paralelo28 -DDEBUG
nvcc cpp_cuda_v29.cu -O3 -o paralelo29 -DDEBUG
nvcc cpp_cuda_v30.cu -O3 -o paralelo30 -DDEBUG
nvcc cpp_cuda_v31.cu -O3 -o paralelo31 -DDEBUG
nvcc cpp_cuda_v32.cu -O3 -o paralelo32 -DDEBUG
echo "Compilação finalizada"

echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq7.txt
echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq8.txt
echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq9.txt

echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo28.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo29.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo30.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo31.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | Delta Minimo |" > paralelo32.txt	  

echo "TOTAL" > sed6.txt

echo "Executando..."
echo "Executando seq7"
for (( i = 0; 5 > i; i++)); do
	./seq_7 nPontos.bin coordenadas.bin >> seq7.txt			
	echo "#"
done
echo "Executando seq8"
for (( i = 0; 5 > i; i++)); do
	./seq_8 nPontos.bin coordenadas.bin >> seq8.txt		
	echo "#"	
done
echo "Executando seq9"
for (( i = 0; 5 > i; i++)); do
	./seq_9 nPontos.bin coordenadas.bin >> seq9.txt	
	echo "#"		
done
echo "Executando sed6"
for (( i = 0; 5 > i; i++)); do
	./sed6 nPontos.bin coordenadas.bin >> sed6.txt	
	echo "#"		
done
echo "Executando paralelo28"
for (( i = 0; 6 > i; i++)); do
	./paralelo28 nPontos.bin coordenadas.bin >> paralelo28.txt	
	echo "#"		
done
echo "Executando paralelo29"
for (( i = 0; 6 > i; i++)); do
	./paralelo29 nPontos.bin coordenadas.bin >> paralelo29.txt
	echo "#"			
done
echo "Executando paralelo30"
for (( i = 0; 6 > i; i++)); do
	./paralelo30 nPontos.bin coordenadas.bin >> paralelo30.txt	
	echo "#"		
done
echo "Executando paralelo31"
for (( i = 0; 6 > i; i++)); do
	./paralelo31 nPontos.bin coordenadas.bin >> paralelo31.txt	
	echo "#"		
done
echo "Executando paralelo32"
for (( i = 0; 6 > i; i++)); do
	./paralelo32 nPontos.bin coordenadas.bin >> paralelo32.txt	
	echo "#"		
done
echo "Execução finalizada"

echo "Gerando médias"

python3 gera_medias.py > medias.txt
