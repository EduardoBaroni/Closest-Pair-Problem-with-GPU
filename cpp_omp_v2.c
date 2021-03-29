// Este programa trata do problema de se achar o par de pontos mais próximos em um plano xy.
// É executado um algoritmo diferente dos encontrados em livros.
// Este programa é baseado no programa sequencial cpp_seq_v2.c e explora paralelismo com OpenMP.
// 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void geraDados(float y, int x);
void Leitura(int *num_pontos, int **X, int **Y, char *argv[]);
float Calculo_Delta_Inicial( int num_pontos, int X[], int Y[] );
float Forca_Bruta(int num_pontos, int num_regioes, int ptsRegiao, float delta_inicial, int X[], int Y[]);
// Ordenação
void MergeSort_BottomUp(int* X, int* Y, int size);
void Merge(int* src1X, int* src1Y, int* src2X, int* src2Y, int len1, int len2, int* destX, int* destY);

int main(int argc, char *argv[])
{
	// Declarações de Variáveis:
	int num_pontos, num_regioes, ptsRegiao = 1024/32;
	int *X, *Y;
	float delta_inicial, delta_minimo;

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	/* 
		Passo 1: Leitura e armazenamento dos pontos. Esse passo é feito lendo um arquivo texto, mas pode-se adaptar para se ler
		arquivos em binário (para casos de testes muito grandes).

	 	Modelo de como o usuário informará os pontos: n m x1 y1 x2 y2 x3 y3... , em que n é a quantidade de pontos,
	  	m é a quantidade de blocos, e xi yi são as coordenadas dos pontos.

	   Teremos dois vetores, X e Y, para armazenar os pontos. Os vetores, juntos, armazenarão os pontos,
		de forma que um ponto no plano xy tem sua parte x armazenada no vetor X, e sua parte Y armazenada no
		vetor Y, sendo que os índices para parte x e a parte y deste ponto deve ser o mesmo. Assim, para um 
		ponto (a,b), se temos (a) armazenado em X[2], então (b) deve estar armazenado em Y[2].
	*/

	#if DEBUG
 		float inicio_leitura = omp_get_wtime();
	#endif

	Leitura(&num_pontos,&X,&Y,argv);
	
	#if DEBUG
		float fim_leitura = omp_get_wtime();
		float leituraTempo = fim_leitura - inicio_leitura;
	#endif
/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/
	
	float start = omp_get_wtime();
	//clock_t inicio = clock();
	// Passo 2: Ordenando os pontos em X:

	#if DEBUG
		float inicio_ordenacao = omp_get_wtime();
	#endif

	MergeSort_BottomUp(X, Y, num_pontos);

	#if DEBUG
		float fim_ordenacao = omp_get_wtime();
		float ordenacaoTempo = fim_ordenacao - inicio_ordenacao;
	#endif
/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	// Passo 3: Calculando o delta inicial (distância euclidiana mínima entre um ponto e seu sucessor armazenado):

	#if DEBUG
		float inicio_calc_distancias = omp_get_wtime();
	#endif
	
	delta_inicial = Calculo_Delta_Inicial(num_pontos, X, Y);
	
	#if DEBUG
		float fim_calc_distancias = omp_get_wtime();
		float distanciasTempo = fim_calc_distancias - inicio_calc_distancias;
	#endif

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	//Passo 4: Dividir os pontos que temos em várias regiões, de forma que cada região tenha aproximadamente a mesma quantidade de pontos.
	
	// Caso não tenhamos quantidade iguais de pontos em todos os blocos, realizamos o tratamento:
	num_regioes = num_pontos / ptsRegiao;	
	if( num_pontos % ptsRegiao != 0 )
		num_regioes += 1;

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	/* 
		Passo 5: Para cada bloco, achar seu delta, utilizando algoritmo de força bruta.

		O algoritmo de força bruta consiste em, testar todos os pares de pontos possíveis,
		a fim de se obter o menor delta possível. 

		OBS: Note que o algoritmo é feito já levando em conta as intersecções entre os blocos.
	*/
	
	#if DEBUG	
		float inicio_forca_bruta = omp_get_wtime();
	#endif

	delta_minimo = Forca_Bruta(num_pontos, num_regioes, ptsRegiao, delta_inicial, X, Y);

	// Imprimindo resultados:
	
	#if DEBUG
		float fim_forca_bruta = omp_get_wtime();
		float forcaBrutaTempo = fim_forca_bruta - inicio_forca_bruta;
	#endif
	
/*-----------------------------------------------------------------------------------------------------------------*/	
/*-----------------------------------------------------------------------------------------------------------------*/
	// Imprimindo resultados

	//clock_t fim = clock();
	//float tempoTotal = (fim - inicio) / (float) CLOCKS_PER_SEC;
	float end = omp_get_wtime();
	float tempoTotal = end - start;
	
	#if DEBUG
		printf("%.5f   |   %.5f      |      %.5f      |      %.5f   |    %.5f   |    %lf   |   %lf\n", 
			leituraTempo, ordenacaoTempo, distanciasTempo, forcaBrutaTempo, tempoTotal, delta_inicial, delta_minimo);
	#else
		printf("Resultado: %lf\n", delta_minimo);
		printf("Tempo    : %.5f\n", tempoTotal);
	#endif

	#if GRAFICO
		geraDados(fim-inicio, num_pontos);
	#endif
	
	return 0;
}

void geraDados(float y, int x)
{
	FILE *y_data = fopen("eixoVertical", "a");	
	fprintf(y_data, "%g", y / (float) CLOCKS_PER_SEC);
	fprintf(y_data, "%s", "\n");
	
	FILE *x_data = fopen("eixoHorizontal", "a");	
	fprintf(x_data, "%d", x);
	fprintf(x_data, "%s", "\n");

	fclose(y_data);
	fclose(x_data);
}

void Leitura(int *num_pontos, int **X, int **Y, char *argv[])
{
	FILE *entrada1, *entrada2;

	entrada1 = fopen(argv[1], "rb");
	entrada2 = fopen(argv[2], "rb");

	// PARA USAR fread: unsigned fread (void *onde_armazenar, int tam_a_ser_lido_em_bytes, int qtd_de_unidades_a_serem_lidas, FILE *fp);
	if (fread( num_pontos, sizeof(int), 1, entrada1))
	{
		*X = (int *) malloc( *num_pontos * sizeof(int) );
		*Y = (int *) malloc( *num_pontos * sizeof(int) );
	}

	// TODO: throw exception
	if (fread( *X, sizeof(int), *num_pontos, entrada2));
	if (fread( *Y, sizeof(int), *num_pontos, entrada2));	

	fclose(entrada1);
	fclose(entrada2);
}

float Calculo_Delta_Inicial( int num_pontos, int X[], int Y[] )
{
	float aux, delta_inicial;

	delta_inicial = (float) INT_MAX;
	long int A,B;

	#pragma omp parallel for reduction(min:delta_inicial) private(aux, A, B)
	for( int i=0 ; i<num_pontos-1; i++ ){

		if( X[i]!=X[i+1] || Y[i]!=Y[i+1] ){

			A = (long int) ( (long int)(X[i]-X[i+1])*(long int)(X[i]-X[i+1]) );
			B = (long int) ( (long int)(Y[i]-Y[i+1])*(long int)(Y[i]-Y[i+1]) );
			
			aux = (float) sqrt( A + B );		

			if( aux < delta_inicial )
				delta_inicial = aux;
		}
	}

	return delta_inicial;
}

float Forca_Bruta(int num_pontos, int num_regioes, int ptsRegiao, float delta_inicial, int X[], int Y[])
{
	float aux, delta_minimo = delta_inicial; // delta_minimo é shared
	int i, j, k, x_final, lim_final;
	long int A, B;
	
	#pragma omp parallel private(x_final, lim_final, j, k, A, B, aux) // Cria região paralela
	{
		float delta_min_privado = delta_minimo; // delta_min_privado é private de cada thread

		#pragma omp for nowait // Distribui iterações do laço pelas threads
		for( i=0 ; i<num_regioes-1 ; i++ ){

			// Cálculo limite final da região i
			x_final = X[ptsRegiao*(i+1)-1];
			lim_final = x_final + (int) delta_minimo;

			for( j=i*ptsRegiao ; j < ((i+1)*ptsRegiao) ; j++ ){

				for( k=j+1 ; X[k]<=lim_final && k<num_pontos ; k++ ){

					// OTIMIZAÇÃO: Olhar a coordenada x
					if(X[k]-X[j]>(int)delta_min_privado ){
						k = num_pontos;
					}
					else if( X[j]!=X[k] || Y[j]!=Y[k] )
					{
						A = (long int) ( (long int)(X[j]-X[k])*(long int)(X[j]-X[k]) );
						B = (long int) ( (long int)(Y[j]-Y[k])*(long int)(Y[j]-Y[k]) );
			
						aux = (float) sqrt( A + B );

						if( aux < delta_min_privado )
						{
							delta_min_privado = aux;
							lim_final = x_final + (int) delta_min_privado;
						}
						
					}
				}
			}
		} // Fim do for i paralelizado, sem barreira

		#pragma omp critical // Cada thread compara seu delta_min_privado com delta_minimo
		if( delta_min_privado < delta_minimo )
		{
			delta_minimo = delta_min_privado;
		}

		#pragma omp barrier
		delta_min_privado = delta_minimo ;

		#pragma omp for nowait // Distribui iterações do laço pelas threads
		for( int j=(num_regioes-1)*ptsRegiao ; j < num_pontos-1 ; j++ ){
		
			for( int k=j+1 ; k<num_pontos ; k++ ){

				if( X[j]!=X[k] || Y[j]!=Y[k] )
				{
					// OTIMIZAÇÃO: Olhar a coordenada x
					if(X[k]-X[j]>(int)delta_min_privado ){
						k = num_pontos;
					}
					else if( X[j]!=X[k] || Y[j]!=Y[k] ){

						A = (long int) ( (long int)(X[j]-X[k])*(long int)(X[j]-X[k]) );
						B = (long int) ( (long int)(Y[j]-Y[k])*(long int)(Y[j]-Y[k]) );
			
						aux = (float) sqrt( A + B );

						if( aux < delta_min_privado )
							delta_min_privado = aux;
					}

				}
			}
		} // Fim do for j paralelizado, sem barreira

		#pragma omp critical // Cada thread compara seu delta_min_privado com delta_minimo
		if( delta_min_privado < delta_minimo )
		{
			delta_minimo = delta_min_privado;
		}
	} // Fim da região paralela: barreira implícita

	return delta_minimo;
}

void MergeSort_BottomUp(int *X, int *Y, int size)
{
	int *tempX = (int*)malloc(sizeof(int) * size);
	int *tempY = (int*)malloc(sizeof(int) * size);
	int *repo1X, *repo1Y, *repo2X, *repo2Y, *auxX, *auxY;
	int stIdx;

	repo1X = X;
	repo1Y = Y;
	repo2X = tempX;
	repo2Y = tempY;

	for (int grpSize = 1; grpSize < size; grpSize <<= 1)
	{
		#pragma omp parallel for
		for (stIdx = 0; stIdx < size; stIdx += 2 * grpSize)
		{
			int nextIdx = stIdx + grpSize;
			int secondGrpSize = MIN(MAX(0, size - nextIdx), grpSize);

			if (secondGrpSize == 0)
			{
				for (int i = 0; i < size - stIdx; i++)
				{
					repo2X[stIdx + i] = repo1X[stIdx + i];
					repo2Y[stIdx + i] = repo1Y[stIdx + i];
				}
			}
			else
			{
				Merge(repo1X + stIdx, repo1Y + stIdx, repo1X + nextIdx, repo1Y + nextIdx,
						grpSize, secondGrpSize, repo2X + stIdx, repo2Y + stIdx);
			}
		}
		auxX = repo1X;
		repo1X = repo2X;
		repo2X = auxX;

		auxY = repo1Y;
		repo1Y = repo2Y;
		repo2Y = auxY;
	}

	if (repo1X != X)
	{
		memcpy(X, tempX, sizeof(int)*size);
		memcpy(Y, tempY, sizeof(int)*size);
	}

	free(tempX);
	free(tempY);
}

void Merge(int* src1X, int* src1Y, int* src2X, int* src2Y, int len1, int len2, int* destX, int* destY)
{
	int idx1 = 0, idx2 = 0, loc = 0;

	while (idx1 < len1 && idx2 < len2)
	{
		if (src1X[idx1] <= src2X[idx2])
		{
			destX[loc] = src1X[idx1];
			destY[loc] = src1Y[idx1];
			idx1++;
		}
		else
		{
			destX[loc] = src2X[idx2];
			destY[loc] = src2Y[idx2];
			idx2++;
		}
		loc++;
	}

	for (int i = idx1; i < len1; i++)
	{
		destX[loc] = src1X[i];
		destY[loc] = src1Y[i];
		loc++;
	}

	for (int i = idx2; i < len2; i++)
	{
		destX[loc] = src2X[i];
		destY[loc] = src2Y[i];
		loc++;
	}
}
