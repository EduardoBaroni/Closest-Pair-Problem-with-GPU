#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(void){
	int num_pontos; // variáveis a serem lidas.
	int minX, maxX, minY, maxY, minZ, maxZ; // Limites do intervalo dos valores
	int aux;
	FILE *saida1, *saida2; // Ponteiro que aponta para o arquivo de saída que será gerado

	printf("Informe os limites min e max em X:\n");
	scanf("%d %d", &minX, &maxX);
	printf("Informe os limites min e max em Y:\n");
	scanf("%d %d", &minY, &maxY);
	printf("Informe os limites min e max em Z:\n");
	scanf("%d %d", &minZ, &maxZ);
	printf("Informe num_pontos:\n");
	scanf("%d", &num_pontos); // Lendo qnt de pts aleatórios a serem gerados
	
	// Abrindo arquivos em modo de leitura binária 
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
