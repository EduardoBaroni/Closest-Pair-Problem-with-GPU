#!/bin/bash

#Variáveis da aplicação

echo "Compilando..."
gcc gerador_bin_v5.c -pedantic -std=c11 -o gerador
gcc cpp_seq_v7.c -pedantic -std=c11 -O3 -o seq_7 -lm
gcc cpp_seq_v8.c -pedantic -std=c11 -O3 -o seq_8 -lm
gcc cpp_seq_v9.c -pedantic -std=c11 -O3 -o seq_9 -lm
nvcc cpp_cuda_v28.cu -O3 -o paralelo28
nvcc cpp_cuda_v29.cu -O3 -o paralelo29
nvcc cpp_cuda_v30.cu -O3 -o paralelo30
nvcc cpp_cuda_v31.cu -O3 -o paralelo31
nvcc cpp_cuda_v32.cu -O3 -o paralelo32
echo "Compilação finalizada"

echo "!   TOTAL   !" > seq7.txt
echo "!   TOTAL   !" > seq8.txt
echo "!   TOTAL   !" > seq9.txt

echo "!   TOTAL   !" > paralelo28.txt
echo "!   TOTAL   !" > paralelo29.txt
echo "!   TOTAL   !" > paralelo30.txt
echo "!   TOTAL   !" > paralelo31.txt
echo "!   TOTAL   !" > paralelo32.txt

for (( i = 0; 5 > i; i++)); do
	./seq_7 nPontos.bin coordenadas.bin >> seq7.txt			
done
for (( i = 0; 5 > i; i++)); do
	./seq_8 nPontos.bin coordenadas.bin >> seq8.txt			
done
for (( i = 0; 5 > i; i++)); do
	./seq_9 nPontos.bin coordenadas.bin >> seq9.txt			
done
for (( i = 0; 5 > i; i++)); do
	./paralelo28 nPontos.bin coordenadas.bin >> paralelo28.txt
done
for (( i = 0; 5 > i; i++)); do
	./paralelo29 nPontos.bin coordenadas.bin >> paralelo28.txt
done
for (( i = 0; 5 > i; i++)); do
	./paralelo30 nPontos.bin coordenadas.bin >> paralelo28.txt
done
for (( i = 0; 5 > i; i++)); do
	./paralelo31 nPontos.bin coordenadas.bin >> paralelo28.txt
done
for (( i = 0; 5 > i; i++)); do
	./paralelo32 nPontos.bin coordenadas.bin >> paralelo28.txt
done

echo "Execução finalizada"