/* 
	Este programa trata do problema de se achar os pares de pontos mais próximos num plano xy. 
	Aqui se utiliza a plataforma CUDA a fim de verificar possíveis soluções eficientes.
*/

// Bibliotecas C
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <limits.h>
#include <time.h>

// Bibliotecas Thrust
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/extrema.h>

// Bibliotecas C++
#include <iostream>
#include <fstream>
#include <iterator>

// Bibliotecas Nvprof
#include "cuda_profiler_api.h"

// Kernels

__global__ void calculaDistancias(unsigned int num_pontos, int *X, int *Y, float *dD)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; // thread corrente

	long int A,B;

	if( index < num_pontos-1 ){


		if( X[index]!=X[index+1] || Y[index]!=Y[index+1] ){

			A = (long int) ( (long int)(X[index] - X[index+1]) * (long int)(X[index] - X[index+1]) );
								
			B = (long int) ( (long int)(Y[index] - Y[index+1]) * (long int)(Y[index] - Y[index+1]) );
		
			dD[index] =  (float) sqrt( (double) (A + B) );

		}
		else{

			dD[index] = INT_MAX;

		}

	}

}

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

__global__ void Forca_Bruta(int num_pontos, int num_regioes, int ptsRegiao, int *X, int *Y, float *Minimos, float delta_inicial)
{

	int i = blockIdx.x; // bloco corrente (coincide com a região corrente)
	int j = blockIdx.x * blockDim.x + threadIdx.x; // thread corrente
	int k; // auxiliar
	float aux, delta_minimo = delta_inicial;
	long int A,B;
	int LimFinal;

	// Passo 5:
	if( i < num_regioes-1 )// Todos as regioes menos a última são tratadas igualmente.
	{ 
		// Calculo do limite final da região
		LimFinal = X[ptsRegiao * (i+1) - 1] + (int) delta_inicial;

		for( k=j+1 ; X[k]<=LimFinal && k<num_pontos ; k++ ){ // cada thread executará esse laço.

			if( X[j] != X[j+1] || Y[j] != Y[j+1] ){

				A = (long int) ( (long int)(X[j] - X[k]) * (long int)(X[j] - X[k]) );
								
				B = (long int) ( (long int)(Y[j] - Y[k]) * (long int)(Y[j] - Y[k]) );
		
				aux =  (float) sqrt( (double) (A + B) );

				if( aux < delta_minimo )
					delta_minimo = aux;
			}
		}
		Minimos[j] = delta_minimo;
	}
	else
	{
		if( j < num_pontos-1 ){

			for( k=j+1 ;  k < num_pontos ; k++ ){ // cada thread executará esse laço.

				if( X[j] != X[j+1] || Y[j] != Y[j+1] ){

					A = (long int) ( (long int)(X[j] - X[k]) * (long int)(X[j] - X[k]) );
								
					B = (long int) ( (long int)(Y[j] - Y[k]) * (long int)(Y[j] - Y[k]) );
		
					aux =  (float) sqrt( (double) (A + B) );

					if( aux < delta_minimo )
						delta_minimo = aux;
				}
			}
			Minimos[j] = delta_minimo;
		}
	}
}

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

// Funções
//PRESTE ATENÇÃO NO '&', significa que se esta passando vector por referência
void leitura(char *argv[], unsigned int *num_pontos, thrust::host_vector<int> &hX, thrust::host_vector<int> &hY)

{
	std::ifstream pts(argv[1], std::ios::binary);
	std::ifstream file(argv[2], std::ios::binary);


	if(file && pts)// Só verificando se não deu falha ao abrir os arquivos
	{
		// Inicializando num_pontos
		pts.read((char*) num_pontos, sizeof(int));

 		// Após obter num pontos pode-se alocar dinamicamente os host vectors
 		hX.resize(*num_pontos);
 		hY.resize(*num_pontos);	
 		
 		// Entao as coordenadas são lidas
		file.read((char*)(hX.data()), hX.size()*sizeof(int));
		file.read((char*)(hY.data()), hY.size()*sizeof(int));

	}

	pts.close();
	file.close();
}

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

int calculaRegioes(unsigned int num_pontos, unsigned int ptsRegiao)
{
	int num_regioes;

	num_regioes = num_pontos / ptsRegiao;	
	
	if( num_pontos % ptsRegiao != 0 )
		num_regioes += 1;
	
	return num_regioes;
}

inline void cpp_medicoes(char* argv[]);

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
	clock_t inicio = clock();
	
/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	#if MEDICAO_INTERMEDIARIA
		cpp_medicoes(argv);
		
	#else	
		// Declaração de variáveis:
		unsigned int num_regioes, num_pontos, ptsRegiao;
		int maxThreadBloco;
		float delta_inicial, delta_minimo;

		// Capturando o máximo número de threads por bloco da máquina
		cudaDeviceGetAttribute(&maxThreadBloco, cudaDevAttrMaxThreadsPerBlock,0);
		ptsRegiao = maxThreadBloco/32;

		// HOST
		thrust::host_vector<int> hX; // Coordenadas x no host
		thrust::host_vector<int> hY; // Coordenadas y no host

		// DEVICE
		thrust::device_vector<int> dX; // Coordenadas x no device
		thrust::device_vector<int> dY; // Coordenadas y no device
		
		// Passo 1: Leitura e armazenamento dos pontos. Esse passo é feito lendo um arquivo binário.

	 	leitura(argv, &num_pontos, hX, hY);

		// Passo 2: Memcpy's do host para device

		dX = hX;
		dY = hY;
		
		// Passo 3: Ordenando os pontos em X:

		thrust::stable_sort_by_key(dX.begin(), dX.end(), dY.begin());

		// Passo 4: Dividir os n pontos que temos em m regioes, de forma que cada bloco tenha aproximadamente a mesma quantidade de pontos.

		num_regioes = calculaRegioes(num_pontos, ptsRegiao);

		//Passo 5: Calculando o delta inicial (distância euclidiana mínima entre um ponto e seu sucessor armazenado):

		thrust::device_vector<float> dD(num_pontos-1); // Vetor de Distâncias (para o delta inicial) no device	
		
		// Forma encontrada de usar vetores da thrust em um kernel: Apontar para cada um deles com novos ponteiros.
		int *X = thrust::raw_pointer_cast(&dX[0]); // aponta para dX
		int *Y = thrust::raw_pointer_cast(&dY[0]); // aponta para dY
		float *d = thrust::raw_pointer_cast(&dD[0]); // aponta para dD

		//Número Máximo de Blocos: 2^31-1 = 2 147 483 647
		int num_blocos;
		
		if( num_pontos % maxThreadBloco != 0 )
			num_blocos = (num_pontos / maxThreadBloco) + 1;
		else
			num_blocos = num_pontos / maxThreadBloco;		

	 	// Kernel que calcula vector de distâncias
		calculaDistancias<<<num_blocos, maxThreadBloco>>>(num_pontos, X, Y, d);
		
	 	cudaDeviceSynchronize();

		thrust::device_vector<float>::iterator iter = thrust::min_element(dD.begin(), dD.end()); // Redução usando thrust para achar delta inicial do vetor de distâncias
		
		delta_inicial = *iter;
		printf("\n\nDelta Inicial: %lf\n\n", delta_inicial);
		
		//Passo 6: Para cada bloco, achar seu delta, utilizando algoritmo de força bruta.

		if( num_regioes%maxThreadBloco != 0 )
			num_blocos = (num_regioes/maxThreadBloco) + 1;
		else
			num_blocos = num_regioes/maxThreadBloco;	

		thrust::device_vector<float> dMin(num_pontos, INT_MAX); // Vetor de minimos
		
		float *Min = thrust::raw_pointer_cast(&dMin[0]); // aponta para dMin 

		Forca_Bruta<<<num_regioes, ptsRegiao>>>(num_pontos, num_regioes, ptsRegiao, X, Y, Min, delta_inicial);
		
		cudaDeviceSynchronize();

		// Redução do vetor dMin:
		iter = thrust::min_element(dMin.begin(), dMin.end());
		
		delta_minimo = *iter;
		printf("Delta mínimo:\n%lf\n", delta_minimo);
	#endif

	clock_t fim = clock();
	printf("Tempo total: %g segundos\n\n", (fim - inicio) / (float) CLOCKS_PER_SEC);

	return 0;
}


inline void cpp_medicoes(char* argv[])
{
	unsigned int num_pontos;
	thrust::host_vector<int> hX;
	thrust::host_vector<int> hY;

	clock_t inicio_leitura = clock();
 	leitura(argv, &num_pontos, hX, hY);
	clock_t fim_leitura = clock();

	printf("\nTempo da função leitura: %g segundos\n\n", (fim_leitura - inicio_leitura) / (float) CLOCKS_PER_SEC);

	clock_t inicio_transferencia = clock();
	thrust::device_vector<int> dX = hX;
	thrust::device_vector<int> dY = hY;
	clock_t fim_transferencia = clock();

	printf("Tempo da transferencia: %g segundos\n\n", (fim_transferencia - inicio_transferencia) / (float) CLOCKS_PER_SEC);

	clock_t inicio_ordenacao = clock();
	thrust::stable_sort_by_key(dX.begin(), dX.end(), dY.begin());
	clock_t fim_ordenacao = clock();

	printf("Tempo da função de ordenação: %g segundos\n\n", (fim_ordenacao - inicio_ordenacao) / (float) CLOCKS_PER_SEC);

	int maxThreadBloco;
	cudaDeviceGetAttribute(&maxThreadBloco, cudaDevAttrMaxThreadsPerBlock,0);
	unsigned int ptsRegiao = maxThreadBloco/32;
	
	unsigned int num_regioes = calculaRegioes(num_pontos, ptsRegiao);

	thrust::device_vector<float> dD(num_pontos-1);
	
	int *X = thrust::raw_pointer_cast(&dX[0]);
	int *Y = thrust::raw_pointer_cast(&dY[0]);
	float *d = thrust::raw_pointer_cast(&dD[0]);

	int num_blocos = num_pontos % maxThreadBloco != 0 ? (num_pontos / maxThreadBloco) + 1 : num_blocos = num_pontos / maxThreadBloco;
	
	clock_t inicio_calc_distancias = clock();
	calculaDistancias<<<num_blocos, maxThreadBloco>>>(num_pontos, X, Y, d);
	
 	cudaDeviceSynchronize();
	clock_t fim_calc_distancias = clock();

	printf("Tempo do kernel Calcula Distâncias: %g segundos\n\n", (fim_calc_distancias - inicio_calc_distancias) / (float) CLOCKS_PER_SEC);

	clock_t inicio_reducao1 = clock();
	thrust::device_vector<float>::iterator iter = thrust::min_element(dD.begin(), dD.end());
	clock_t fim_reducao1 = clock();

	float delta_inicial = *iter;

	printf("\n\nDelta Inicial: %lf\n\n", delta_inicial);
	printf("Tempo da redução1: %g segundos\n\n", (fim_reducao1 - inicio_reducao1) / (float) CLOCKS_PER_SEC);

	num_blocos = num_regioes % maxThreadBloco != 0 ? (num_regioes / maxThreadBloco) + 1 : num_blocos = num_regioes / maxThreadBloco;

	thrust::device_vector<float> dMin(num_pontos, INT_MAX);
	float *Min = thrust::raw_pointer_cast(&dMin[0]);

	clock_t inicio_forca_bruta = clock();
	Forca_Bruta<<<num_regioes, ptsRegiao>>>(num_pontos, num_regioes, ptsRegiao, X, Y, Min, delta_inicial);
	
	cudaDeviceSynchronize();
	clock_t fim_forca_bruta = clock();

	printf("Tempo do kernel Força Bruta: %g segundos\n\n", (fim_forca_bruta - inicio_forca_bruta) / (float) CLOCKS_PER_SEC);
	
	clock_t inicio_reducao2 = clock();
	iter = thrust::min_element(dMin.begin(), dMin.end());
	clock_t fim_reducao2 = clock();

	float delta_minimo = *iter;

	printf("Delta mínimo:\n%lf\n", delta_minimo);
	printf("Tempo da redução2: %g segundos\n\n", (fim_reducao2 - inicio_reducao2) / (float) CLOCKS_PER_SEC);
}