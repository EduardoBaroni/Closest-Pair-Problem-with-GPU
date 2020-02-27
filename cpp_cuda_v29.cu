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

#include <cuda_profiler_api.h>


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


			// OTIMIZAÇÃO: Olhar a coordenada x
			if(X[k]-X[j]>(int)delta_minimo ){
				k = num_pontos;
			}
			else if( X[j]!=X[k] || Y[j]!=Y[k] ){

				A = (long int) ( (long int)(X[j]-X[k])*(long int)(X[j]-X[k]) );
			
				B = (long int) ( (long int)(Y[j]-Y[k])*(long int)(Y[j]-Y[k]) );
	
				aux = (float) sqrt( (double) (A + B) );

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

				// OTIMIZAÇÃO: Olhar a coordenada x
				if(X[k]-X[j]>(int)delta_minimo ){
					k = num_pontos;
				}
				else if( X[j]!=X[k] || Y[j]!=Y[k] ){

					A = (long int) ( (long int)(X[j]-X[k])*(long int)(X[j]-X[k]) );
				
					B = (long int) ( (long int)(Y[j]-Y[k])*(long int)(Y[j]-Y[k]) );
		
					aux = (float) sqrt( (double) (A + B) );

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

void geraDados(float y, int x)
{
	FILE *y_data = fopen("eixoVertical_cuda", "a");	
	fprintf(y_data, "%g", y / (float) CLOCKS_PER_SEC);
	fprintf(y_data, "%s", "\n");
	
	FILE *x_data = fopen("eixoHorizontal_cuda", "a");	
	fprintf(x_data, "%d", x);
	fprintf(x_data, "%s", "\n");

	fclose(y_data);
	fclose(x_data);
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

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
	// Declaração de variáveis:
	unsigned int num_regioes, num_pontos, ptsRegiao;
	int maxThreadBloco;
	float delta_inicial;

	// Capturando o máximo número de threads por bloco da máquina
	cudaDeviceGetAttribute(&maxThreadBloco, cudaDevAttrMaxThreadsPerBlock, 0);
	ptsRegiao = maxThreadBloco/32;

	// HOST
	thrust::host_vector<int> hX; // Coordenadas x no host
	thrust::host_vector<int> hY; // Coordenadas y no host

	// DEVICE
	thrust::device_vector<int> dX; // Coordenadas x no device
	thrust::device_vector<int> dY; // Coordenadas y no device

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	// Passo 1: Leitura e armazenamento dos pontos. Esse passo é feito lendo um arquivo binário.

	#if DEBUG
		clock_t inicio_leitura = clock();
 	#endif

 	leitura(argv, &num_pontos, hX, hY);
	
 	#if DEBUG
		clock_t fim_leitura = clock();
		float leituraTempo = (fim_leitura - inicio_leitura) / (float) CLOCKS_PER_SEC;
	#endif

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/
	clock_t inicio = clock();
	// Passo 2: Memcpy's do host para device

	#if DEBUG
		clock_t inicio_transferencia = clock();
	#endif

	dX = hX;
	dY = hY;
	
	#if DEBUG
		clock_t fim_transferencia = clock();
		float transfTempo = (fim_transferencia - inicio_transferencia) / (float) CLOCKS_PER_SEC;
	#endif

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	// Passo 3: Ordenando os pontos em X:

	#if DEBUG
		clock_t inicio_ordenacao = clock();
	#endif

	thrust::sort_by_key(dX.begin(), dX.end(), dY.begin());
	
	#if DEBUG
		clock_t fim_ordenacao = clock();
		float ordenacaoTempo = (fim_ordenacao - inicio_ordenacao) / (float) CLOCKS_PER_SEC;
	#endif

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	// Passo 4: Dividir os n pontos que temos em m regioes, de forma que cada bloco tenha aproximadamente a mesma quantidade de pontos.

	num_regioes = calculaRegioes(num_pontos, ptsRegiao);

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	//Passo 5: Calculando o delta inicial (distância euclidiana mínima entre um ponto e seu sucessor armazenado):
 	
	thrust::device_vector<float> dD(num_pontos-1); // Vetor de Distâncias (para o delta inicial) no device	

	// Forma encontrada de usar vetores da thrust em um kernel: Apontar para cada um deles com novos ponteiros.
	int *X = thrust::raw_pointer_cast(&dX[0]); // aponta para dX
	int *Y = thrust::raw_pointer_cast(&dY[0]); // aponta para dY
	float *d = thrust::raw_pointer_cast(&dD[0]); // aponta para dD

	//Número Máximo de Blocos: 2^31-1 = 2 147 483 647
	int num_blocos = num_pontos % maxThreadBloco != 0 ? (num_pontos / maxThreadBloco) + 1 : num_pontos / maxThreadBloco;

	#if DEBUG
		clock_t inicio_calc_distancias = clock();
	#endif
 	// Kernel que calcula vector de distâncias
	calculaDistancias<<<num_blocos, maxThreadBloco>>>(num_pontos, X, Y, d);
	
 	cudaDeviceSynchronize(); // Necessário

 	#if DEBUG
		clock_t fim_calc_distancias = clock();
		float distanciasTempo = (fim_calc_distancias - inicio_calc_distancias) / (float) CLOCKS_PER_SEC;
	
		clock_t inicio_reducao1 = clock();
	#endif

	// Redução usando thrust para achar delta inicial do vetor de distâncias
	thrust::device_vector<float>::iterator iter = thrust::min_element(dD.begin(), dD.end()); 
	delta_inicial = *iter;

	#if DEBUG
		clock_t fim_reducao1 = clock();
		float reducao1Tempo = (fim_reducao1 - inicio_reducao1) / (float) CLOCKS_PER_SEC;	
	#endif

/*-----------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------*/

	//Passo 6: Para cada bloco, achar seu delta, utilizando algoritmo de força bruta.

	num_blocos = num_regioes%maxThreadBloco != 0 ? (num_regioes/maxThreadBloco) + 1 : num_regioes/maxThreadBloco;

	thrust::device_vector<float> dMin(num_pontos, INT_MAX); // Vetor de minimos
	float *Min = thrust::raw_pointer_cast(&dMin[0]); // aponta para dMin 

	#if DEBUG
		clock_t inicio_forca_bruta = clock();
	#endif

	cudaProfilerStart();

	Forca_Bruta<<<num_regioes, ptsRegiao>>>(num_pontos, num_regioes, ptsRegiao, X, Y, Min, delta_inicial);
	
	cudaDeviceSynchronize();

	cudaProfilerStop(); 
	
	#if DEBUG
		clock_t fim_forca_bruta = clock();
		float forcaBrutaTempo = (fim_forca_bruta - inicio_forca_bruta) / (float) CLOCKS_PER_SEC;

		clock_t inicio_reducao2 = clock();
	#endif
	// Redução do vetor dMin
	thrust::device_vector<float>::iterator iter2 = thrust::min_element(dMin.begin(), dMin.end());

	#if DEBUG
		float delta_minimo = *iter2;

		clock_t fim_reducao2 = clock();
		float reducao2Tempo = (fim_reducao2 - inicio_reducao2) / (float) CLOCKS_PER_SEC;
	#endif

/*-----------------------------------------------------------------------------------------------------------------*/	
/*-----------------------------------------------------------------------------------------------------------------*/
	//Imprimindo resultados

	clock_t fim = clock();
	float tempoTotal = (fim - inicio) / (float) CLOCKS_PER_SEC;
	
	#if DEBUG
		printf("%.5f      %.5f          %.5f            %.5f        %.5f        %.5f       %.5f    %.5f       %lf      %lf\n", 
			leituraTempo, transfTempo, ordenacaoTempo, distanciasTempo, reducao1Tempo, forcaBrutaTempo, reducao2Tempo, tempoTotal, 
			delta_inicial, delta_minimo);
	#else
		printf("   %.5f\n", tempoTotal);
	#endif
	
	return 0;
}
