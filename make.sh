#!/bin/bash

if [[ ! -f "$gerador" || "$seq0" || "$seq1" || "$seq2" || "$paralelo0" || "$paralelo1" || "$paralelo2" || "$paralelo3" ]]; then
	echo "Compilando..."
	gcc gerador.c -pedantic -std=c11 -o gerador -DSCRIPT_MODE
	gcc cpp_seq_v7.c -pedantic -std=c11 -O3 -o seq0 -lm
	gcc cpp_seq_v8.c -pedantic -std=c11 -O3 -o seq1 -lm
	gcc cpp_seq_v9.c -pedantic -std=c11 -O3 -o seq2 -lm
	gcc sed6_quick.c -pedantic -std=c11 -O3 -o sed6 -lm
	nvcc cpp_cuda_v0.cu -O3 -o paralelo0
	nvcc cpp_cuda_v1.cu -O3 -o paralelo1
	nvcc cpp_cuda_v2.cu -O3 -o paralelo2
	nvcc cpp_cuda_v3.cu -O3 -o paralelo3
	echo "Compilação finalizada"
fi

echo "!   TOTAL   !" > seq0.txt
echo "!   TOTAL   !" > seq1.txt
echo "!   TOTAL   !" > seq2.txt
echo "|   TOTAL   |" > sed6.txt

echo "!   TOTAL   !" > paralelo0.txt
echo "!   TOTAL   !" > paralelo1.txt
echo "!   TOTAL   !" > paralelo2.txt
echo "!   TOTAL   !" > paralelo3.txt

echo "Executando..."

for (( i = 0; 10 > i; i++)); do
	echo "Executando seq0"
	./seq0 nPontos.bin coordenadas.bin >> seq0.txt			

	echo "Executando seq1"
	./seq1 nPontos.bin coordenadas.bin >> seq1.txt			

	echo "Executando seq2"
	./seq2 nPontos.bin coordenadas.bin >> seq2.txt			

	echo "Executando sed6"
	./sed6 nPontos.bin coordenadas.bin >> sed6.txt			

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
done

echo "Execução finalizada"
