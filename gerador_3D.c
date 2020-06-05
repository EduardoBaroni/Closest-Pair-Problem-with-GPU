#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	int num_pontos = atoi(argv[1]), // variáveis a serem lidas.
		minX = atoi(argv[2]),
		maxX = atoi(argv[3]),
		minY = atoi(argv[4]),
		maxY = atoi(argv[5]),
		minZ = atoi(argv[6]),
		maxZ = atoi(argv[7]); // Limites do intervalo dos valores

	int aux;
	FILE *saida1, *saida2; // Ponteiro que aponta para o arquivo de saída que será gerado

	// Abrindo arquivos em modo de escrita binária 
	// Obs.: Esse modo cria o arquivo caso ele não exista :)
	saida1 = fopen("nPontos.bin", "wb"); // Agora se chama nPontos.bin
	saida2 = fopen("coordenadas.bin", "wb"); // Agora se chama coordenadas.bin

	if(saida1 == NULL || saida2 == NULL){
		printf("Erro ao ler o arquivo.\n");
		return 0; 
	}

	// Escrevendo: num de pontos
	fwrite(&num_pontos, sizeof(int), 1, saida1); // Escrevendo número de pontos no arquivo.

	srand(time(NULL)); // semente

	// Laço gerador das coordenadas para serem escritas no arquivo. Em X, Y e Z teremos intervalos informados pelo usuário.
	// Primeiro são gerados as coordenadas em X:
	for(int i = 0; i < num_pontos ; i++){

		aux = minX + ( rand() % (maxX-minX) );
		fwrite(&aux , sizeof(int), 1, saida2);

		//printf("Valor aleatório em X: %d\n", aux);
	}

	// Em seguida em Y:
	for(int i = 0; i < num_pontos ; i++){
		
		aux = minY + ( rand() % (maxY-minY) );
		fwrite(&aux , sizeof(int), 1, saida2);

		//printf("Valor aleatório em Y: %d\n", aux);
	}			

	// E por fim em Z:
	for(int i = 0; i < num_pontos ; i++){
		
		aux = minZ + ( rand() % (maxZ-minZ) );
		fwrite(&aux , sizeof(int), 1, saida2);

		//printf("Valor aleatório em Z: %d\n", aux);
	}			

	fclose(saida1);
	fclose(saida2);

	return 0;
}
