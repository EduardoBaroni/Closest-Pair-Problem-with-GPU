// Este programa trata do problema de se achar o par de pontos mais próximos em um plano xy.
// É executado um algoritmo diferente dos encontrados em livros.
// Este programa é totalmente sequencial e abordagens com paralelismo serão tratadas em outros programas.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

void Leitura(int *num_pontos, int **X, int **Y, char *argv[]);
float Calculo_Delta_Inicial( int num_pontos, int X[], int Y[] );
float Forca_Bruta(int num_pontos, int num_regioes, int ptsRegiao, float delta_inicial, int X[], int Y[]);
// Ordenação
void troca( int *a, int *b);
void quicksort(int p, int r, int X[], int Y[], int V[]);
int separa(int p, int r, int X[], int Y[], int V[]);

int main(int argc, char *argv[])
{
	// Declarações de Variáveis:
	clock_t inicio = clock();
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

 	clock_t inicio_leitura = clock();
	Leitura(&num_pontos,&X,&Y,argv);
	clock_t fim_leitura = clock();
	
	printf("\nTempo da função leitura: %g segundos\n\n", (fim_leitura - inicio_leitura) / (float) CLOCKS_PER_SEC);

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	// Passo 2: Ordenando os pontos em X:

	clock_t inicio_ordenacao = clock();

	// Vetor auxiliar de posições: (Para tornar o quicksort estável)
	int *X_position = (int *) malloc( num_pontos*sizeof(int) );
	for( int i=0 ; i<num_pontos ; i++ )
		X_position[i] = i;


	// Chamada do quicksort (esta ordenação é instável)
	quicksort(0, num_pontos-1, X, Y, X_position);


	// Fazer uma varredura para garantir a estabilidade.
	for( int i=0, j ; i<num_pontos-1 ; i++ ){
		if( X[i] == X[i+1] ){
			for( j=i+2 ; X[j]==X[i] ; j++ );
			quicksort( i, j-1 , X_position, Y, X);
			i=j-1;
		}
	}
	free(X_position);


	clock_t fim_ordenacao = clock();

	printf("Tempo da função de ordenação: %g segundos\n\n", (fim_ordenacao - inicio_ordenacao) / (float) CLOCKS_PER_SEC);

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	// Passo 3: Calculando o delta inicial (distância euclidiana mínima entre um ponto e seu sucessor armazenado):

	clock_t inicio_calc_distancias = clock();
	delta_inicial = Calculo_Delta_Inicial(num_pontos, X, Y);
	clock_t fim_calc_distancias = clock();

	printf("Tempo da função Calcula Distâncias: %g segundos\n\n", (fim_calc_distancias - inicio_calc_distancias) / (float) CLOCKS_PER_SEC);

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
	
	clock_t inicio_forca_bruta = clock();
	delta_minimo = Forca_Bruta(num_pontos, num_regioes, ptsRegiao, delta_inicial, X, Y);

	// Imprimindo resultados:
	printf("\nDelta mínimo:\n%lf\n", delta_minimo);

	clock_t fim_forca_bruta = clock();

	printf("Tempo da função Força Bruta: %g segundos\n\n", (fim_forca_bruta - inicio_forca_bruta) / (float) CLOCKS_PER_SEC);

	clock_t fim = clock();

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	printf("Tempo total: %g segundos\n\n", (fim - inicio) / (float) CLOCKS_PER_SEC);

	return 0;
}


void Leitura(int *num_pontos, int **X, int **Y, char *argv[])
{
	FILE *entrada1, *entrada2;

	entrada1 = fopen(argv[1], "rb");
	entrada2 = fopen(argv[2], "rb");

	// PARA USAR fread: unsigned fread (void *onde_armazenar, int tam_a_ser_lido_em_bytes, int qtd_de_unidades_a_serem_lidas, FILE *fp);
	fread( num_pontos, sizeof(int), 1, entrada1);

	*X = (int *) malloc( *num_pontos * sizeof(int) );
	*Y = (int *) malloc( *num_pontos * sizeof(int) );

	fread( *X, sizeof(int), *num_pontos, entrada2);
	fread( *Y, sizeof(int), *num_pontos, entrada2);

	fclose(entrada1);
	fclose(entrada2);
}

float Calculo_Delta_Inicial( int num_pontos, int X[], int Y[] )
{
	float aux, delta_inicial;

	delta_inicial = (float) INT_MAX;
	long int A,B;

	for( int i=0 ; i<num_pontos-1; i++ ){

		if( X[i]!=X[i+1] || Y[i]!=Y[i+1] ){

			A = (long int) ( (long int)(X[i]-X[i+1])*(long int)(X[i]-X[i+1]) );
			B = (long int) ( (long int)(Y[i]-Y[i+1])*(long int)(Y[i]-Y[i+1]) );
			
			aux = (float) sqrt( A + B );
			
		}

		

		if( aux < delta_inicial )
			delta_inicial = aux;
	}

	printf("\n\nDelta Inicial: %lf\n\n", delta_inicial);

	return delta_inicial;
}

float Forca_Bruta(int num_pontos, int num_regioes, int ptsRegiao, float delta_inicial, int X[], int Y[])
{
	float aux, delta_minimo = delta_inicial;
	int i;
	long int A,B;
	int lim_final;
	
	for( i=0 ; i<num_regioes-1 ; i++ ){

		// Cálculo limite final da região i
		lim_final = X[ptsRegiao*(i+1)-1] + (int) delta_minimo;

		for( int j=i*ptsRegiao ; j < ((i+1)*ptsRegiao) ; j++ ){

			for( int k=j+1 ; X[k]<=lim_final && k<num_pontos ; k++ ){

				if( X[j]!=X[k] || Y[j]!=Y[k] ){

					A = (long int) ( (long int)(X[j]-X[k])*(long int)(X[j]-X[k]) );
					B = (long int) ( (long int)(Y[j]-Y[k])*(long int)(Y[j]-Y[k]) );
			
					aux = (float) sqrt( A + B );

					if( aux < delta_minimo )
						delta_minimo = aux;
				}

			}
		}
	}

	for( int j=i*ptsRegiao ; j < num_pontos-1 ; j++ ){
	
		for( int k=j+1 ; k<num_pontos ; k++ ){

			if( X[j]!=X[k] || Y[j]!=Y[k] ){

				A = (long int) ( (long int)(X[j]-X[k])*(long int)(X[j]-X[k]) );
				B = (long int) ( (long int)(Y[j]-Y[k])*(long int)(Y[j]-Y[k]) );
			
				aux = (float) sqrt( A + B );

				if( aux < delta_minimo )
					delta_minimo = aux;
			}
		}
	}

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
int separa(int p, int r, int X[], int Y[], int V[])
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
		}
		else
			return j;
	}
}

/* Recebe um vetor v[p..r-1] e o rearranja em ordem crescente */
void quicksort(int p, int r, int X[], int Y[], int V[])
{
	int q;
	if (p < r) {
		q = separa(p, r, X, Y, V);
		quicksort(p, q, X, Y, V);
		quicksort(q+1, r, X, Y, V);
	}
}

