#!/bin/bash

#Variáveis da aplicação

echo "Compilando..."
!gcc gerador_bin_v5.c -pedantic -std=c11 -o gerador
!gcc cpp_seq_v7.c -pedantic -std=c11 -O3 -o seq_7 -lm -DDEBUG
!gcc cpp_seq_v8.c -pedantic -std=c11 -O3 -o seq_8 -lm -DDEBUG
!gcc cpp_seq_v9.c -pedantic -std=c11 -O3 -o seq_9 -lm -DDEBUG
!nvcc cpp_cuda_v28.cu -O3 -o paralelo28 -DDEBUG
!nvcc cpp_cuda_v29.cu -O3 -o paralelo29 -DDEBUG
!nvcc cpp_cuda_v30.cu -O3 -o paralelo30 -DDEBUG
!nvcc cpp_cuda_v31.cu -O3 -o paralelo31 -DDEBUG
!nvcc cpp_cuda_v32.cu -O3 -o paralelo32 -DDEBUG
echo "Compilação finalizada"

echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq7.txt
echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq8.txt
echo "Comparações | Leitura |  Ordenação  | Calcula Delta Incial |  Força Bruta  |   TOTAL   || Delta Inicial | Delta Minimo |" > seq9.txt

echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | 
	  Delta Minimo |" > paralelo28.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | 
	  Delta Minimo |" > paralelo29.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | 
	  Delta Minimo |" > paralelo30.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | 
	  Delta Minimo |" > paralelo31.txt
echo "Leitura | Transferência |  Ordenação  | Calcula Delta Incial | Redução 1 |  Força Bruta  | Redução 2 |   TOTAL   || Delta Inicial | 
	  Delta Minimo |" > paralelo32.txt	  

for (( i = 0; 6 > i; i++)); do
	!./seq_7 nPontos.bin coordenadas.bin >> seq7.txt			
done
for (( i = 0; 6 > i; i++)); do
	!./seq_8 nPontos.bin coordenadas.bin >> seq8.txt			
done
for (( i = 0; 6 > i; i++)); do
	!./seq_9 nPontos.bin coordenadas.bin >> seq9.txt			
done
for (( i = 0; 6 > i; i++)); do
	!./paralelo28 nPontos.bin coordenadas.bin >> paralelo28.txt			
done
for (( i = 0; 6 > i; i++)); do
	!./paralelo29 nPontos.bin coordenadas.bin >> paralelo29.txt			
done
for (( i = 0; 6 > i; i++)); do
	!./paralelo30 nPontos.bin coordenadas.bin >> paralelo30.txt			
done
for (( i = 0; 6 > i; i++)); do
	!./paralelo31 nPontos.bin coordenadas.bin >> paralelo31.txt			
done
for (( i = 0; 6 > i; i++)); do
	!./paralelo32 nPontos.bin coordenadas.bin >> paralelo32.txt			
done

echo "Execução finalizada"