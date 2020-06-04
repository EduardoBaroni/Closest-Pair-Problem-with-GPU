// Este programa trata do problema de se achar o par de pontos mais próximos em um plano xyz.
// É executado um algoritmo diferente dos encontrados em livros.
// Este programa é totalmente sequencial e abordagens com paralelismo serão tratadas em outros programas.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

void Leitura(int *num_pontos, int **X, int **Y, int **Z, char *argv[]);
float Calculo_Delta_Inicial( int num_pontos, int X[], int Y[], int Z[]);
float Forca_Bruta(int num_pontos, int num_regioes, int ptsRegiao, float delta_inicial, int X[], int Y[], int Z[]);
void troca( int *a, int *b);
void quicksort(int p, int r, int X[], int Y[], int Z[], int V[]);
int separa(int p, int r, int X[], int Y[], int Z[], int V[]);

int main(int argc, char *argv[])
{
	// Declarações de Variáveis:
	int num_pontos, num_regioes, ptsRegiao = 1024/32;
	int *X, *Y, *Z;
	float delta_inicial, delta_minimo;

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	/* 
		Passo 1: Leitura e armazenamento dos pontos. Esse passo é feito lendo dois arquivos binários, o primeiro
		relativo ao número de pontos, e o segundo com as coordenadas.

	 	Modelo de como o usuário informará os pontos: x1 y1 x2 y2 x3 y3...

	   Teremos três vetores, X, Y e Z, para armazenar os pontos. Os vetores, juntos, armazenarão os pontos,
		de forma que um ponto no plano xyz tem sua parte x armazenada no vetor X, e sua parte Y armazenada no
		vetor Y e assim por diante, sendo que os índices para parte x, parte y e parte z deste ponto deve ser o mesmo.
	*/

	#if DEBUG // medir tempo da leitura
 		clock_t inicio_leitura = clock();
	#endif

	Leitura(&num_pontos,&X,&Y,&Z,argv);
	
	#if DEBUG
		clock_t fim_leitura = clock();
		printf("\nTempo da função leitura: %g segundos\n\n", (fim_leitura - inicio_leitura) / (float) CLOCKS_PER_SEC);
	#endif

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/
	
	clock_t inicio = clock(); // medição do tempo total do programa
	// Passo 2: Ordenando os pontos em X:

	#if DEBUG // medir tempo da ordenação
		clock_t inicio_ordenacao = clock();
	#endif

	// Vetor auxiliar de posições: (Para tornar o quicksort estável)
	int *X_position = (int *) malloc( num_pontos*sizeof(int) );
	for( int i=0 ; i<num_pontos ; i++ )
		X_position[i] = i;

	// Chamada do quicksort (esta ordenação é instável)
	quicksort(0, num_pontos-1, X, Y, Z, X_position);

	// Fazer uma varredura para garantir a estabilidade.
	for( int i=0, j ; i<num_pontos-1 ; i++ ){
		if( X[i] == X[i+1] ){
			for( j=i+2 ; X[j]==X[i] ; j++ );
			quicksort( i, j-1 , X_position, Y, Z, X);
			i=j-1;
		}
	}
	free(X_position);

	#if DEBUG
		clock_t fim_ordenacao = clock();
		printf("Tempo da função de ordenação: %g segundos\n\n", (fim_ordenacao - inicio_ordenacao) / (float) CLOCKS_PER_SEC);
	#endif
/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/
	
	// Passo 3: Calculando o delta inicial (distância euclidiana mínima entre um ponto e seu sucessor armazenado):

	#if DEBUG // medir tempo do calculo do delta_inicial
		clock_t inicio_calc_distancias = clock();
	#endif
	
	delta_inicial = Calculo_Delta_Inicial(num_pontos, X, Y, Z);
	
	#if DEBUG
		clock_t fim_calc_distancias = clock();
		printf("Tempo da função Calcula Distâncias: %g segundos\n\n", (fim_calc_distancias - inicio_calc_distancias) / (float) CLOCKS_PER_SEC);
	#endif

	printf("\n\nDelta Inicial: %lf\n\n", delta_inicial);
/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/
   
	//Passo 4: Dividir os pontos que temos em várias regiões, de forma que cada região tenha aproximadamente a mesma quantidade de pontos.
	// Caso não tenhamos quantidade iguais de pontos em todos os blocos, realizamos um tratamento.

	num_regioes = num_pontos / ptsRegiao;	
	if( num_pontos % ptsRegiao != 0 )
		num_regioes += 1;

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/
	
	/* 
		Passo 5: Para cada bloco, achar seu delta, utilizando algoritmo de força bruta.
		O algoritmo de força bruta consiste em, testar todos os pares de pontos possíveis,
		a fim de se obter o menor delta possível. 
		OBS: Note que o algoritmo é feito já levando em conta as intersecções entre os blocos (delta_inicial).
	*/

	#if DEBUG // medir tempo do força bruta
		clock_t inicio_forca_bruta = clock();
	#endif

	delta_minimo = Forca_Bruta(num_pontos, num_regioes, ptsRegiao, delta_inicial, X, Y, Z);

	// Imprimindo resultados:

	#if DEBUG
		clock_t fim_forca_bruta = clock();
		printf("Tempo da função Força Bruta: %g segundos\n\n", (fim_forca_bruta - inicio_forca_bruta) / (float) CLOCKS_PER_SEC);
	#endif
	printf("\nDelta mínimo:\n%lf\n", delta_minimo);
		
	clock_t fim = clock();
/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	// Impressão do resultado final.
	printf("\n\nTempo total: %g segundos\n\n", (fim - inicio) / (float) CLOCKS_PER_SEC);

	return 0;
}

void Leitura(int *num_pontos, int **X, int **Y, int **Z, char *argv[])
{
	FILE *entrada1, *entrada2;

	entrada1 = fopen(argv[1], "rb");
	entrada2 = fopen(argv[2], "rb");

	// PARA USAR fread: unsigned fread (void *onde_armazenar, int tam_a_ser_lido_em_bytes, int qtd_de_unidades_a_serem_lidas, FILE *fp);
	if (fread( num_pontos, sizeof(int), 1, entrada1))
	{
		*X = (int *) malloc( *num_pontos * sizeof(int) );
		*Y = (int *) malloc( *num_pontos * sizeof(int) );
		*Z = (int *) malloc( *num_pontos * sizeof(int) );
	}

	if(fread( *X, sizeof(int), *num_pontos, entrada2));
	if(fread( *Y, sizeof(int), *num_pontos, entrada2));	
	if(fread( *Z, sizeof(int), *num_pontos, entrada2));	

	fclose(entrada1);
	fclose(entrada2);
}

float Calculo_Delta_Inicial( int num_pontos, int X[], int Y[], int Z[] )
{
	float aux, delta_inicial;

	delta_inicial = (float) INT_MAX;
	long int A,B,C;

	for( int i=0 ; i<num_pontos-1; i++ ){

		if( X[i]!=X[i+1] || Y[i]!=Y[i+1] || Z[i]!=Z[i+1] ){ // se não forem pontos coincidentes faça

			A = (long int) ( (long int)(X[i]-X[i+1])*(long int)(X[i]-X[i+1]) );
			B = (long int) ( (long int)(Y[i]-Y[i+1])*(long int)(Y[i]-Y[i+1]) );
			C = (long int) ( (long int)(Z[i]-Z[i+1])*(long int)(Z[i]-Z[i+1]) );
			
			aux = (float) sqrt( A + B + C );		
		}

		if( aux < delta_inicial )
			delta_inicial = aux;
	}

	return delta_inicial;
}

float Forca_Bruta(int num_pontos, int num_regioes, int ptsRegiao, float delta_inicial, int X[], int Y[], int Z[])
{
	float aux, delta_minimo = delta_inicial;
	int i;
	long int A,B,C;
	int lim_final, x_final;
	
	#if CONTADOR // conta número de calculos de distância euclidiana
		int cont = 0;
	#endif

	for( i=0 ; i<num_regioes-1 ; i++ ){

		// Cálculo limite final da região i
		x_final = X[ptsRegiao*(i+1)-1];
		lim_final = x_final + (int) delta_minimo;

		for( int j=i*ptsRegiao ; j < ((i+1)*ptsRegiao) ; j++ ){

			for( int k=j+1 ; X[k]<=lim_final && k<num_pontos ; k++ ){

				// OTIMIZAÇÃO: Olhar a coordenada x
				if(X[k]-X[j]>(int)delta_minimo ){
					k = num_pontos;
				}
				else if( X[j]!=X[k] || Y[j]!=Y[k] || Z[j]!=Z[k] ) // se não forem pontos coincidentes faça
				{
					#if CONTADOR
						cont++;
					#endif

					A = (long int) ( (long int)(X[j]-X[k])*(long int)(X[j]-X[k]) );
					B = (long int) ( (long int)(Y[j]-Y[k])*(long int)(Y[j]-Y[k]) );
					C = (long int) ( (long int)(Z[j]-Z[k])*(long int)(Z[j]-Z[k]) );
		
					aux = (float) sqrt( A + B + C );

					if( aux < delta_minimo )
					{
						delta_minimo = aux;
						lim_final = x_final + (int) delta_minimo;
					}
					
				}
			}
		}
	}

	for( int j=i*ptsRegiao ; j < num_pontos-1 ; j++ ){
	
		for( int k=j+1 ; k<num_pontos ; k++ ){

			if( X[j]!=X[k] || Y[j]!=Y[k] || Z[j]!=Z[k] ) // se não forem pontos coincidentes faça
			{
				#if CONTADOR
					cont++;
				#endif

				// OTIMIZAÇÃO: Olhar a coordenada x
				if(X[k]-X[j]>(int)delta_minimo ){
					k = num_pontos;
				}
				else if( X[j]!=X[k] || Y[j]!=Y[k] ){

					A = (long int) ( (long int)(X[j]-X[k])*(long int)(X[j]-X[k]) );
					B = (long int) ( (long int)(Y[j]-Y[k])*(long int)(Y[j]-Y[k]) );
					C = (long int) ( (long int)(Z[j]-Z[k])*(long int)(Z[j]-Z[k]) );
		
					aux = (float) sqrt( A + B + C );

					if( aux < delta_minimo )
						delta_minimo = aux;
				}

			}
		}
	}

	#if CONTADOR
		printf("Distâncias calculadas = %d\n", cont);
	#endif

	return delta_minimo;
}

// Troca dois valores por meio de um auxiliar.
void troca( int *a, int *b )
{
	int aux;

	aux = *a;
	*a = *b;
	*b = aux;
}

/* Recebe um par de números inteiros p e r, com p <= r e um vetor X[p..r]
de números inteiros e rearranja seus elementos e devolve um número inteiro j em p..r tal que X[p..j-1] <= X[j] < X[j+1..r] */
int separa(int p, int r, int X[], int Y[], int Z[], int V[])
{
	int i, j;
	int x;
	
	x = X[p];
	i = p - 1;
	j = r + 1;
	while (1) {
		do {
			j--;
		} while (X[j] > x);
		do {
				i++;
		} while (X[i] < x);
		if (i < j){
			troca(&X[i], &X[j]);
			troca(&V[i], &V[j]);
			troca(&Y[i], &Y[j]);
			troca(&Z[i], &Z[j]);
		}
		else
			return j;
	}
}

/* Recebe um vetor v[p..r-1] e o rearranja em ordem crescente */
void quicksort(int p, int r, int X[], int Y[], int Z[], int V[])
{
	int q;
	if (p < r) {
		q = separa(p, r, X, Y, Z, V);
		quicksort(p, q, X, Y, Z, V);
		quicksort(q+1, r, X, Y, Z, V);
	}
}

