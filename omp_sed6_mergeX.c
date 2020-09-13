/* Algoritmo - Sedgewick */
/* Quinta versao, para pontos 2D */
/*
Melhorias:
-uniu-se o mergesort com o closest pair para diminuir lacos
*/

#ifdef _MSC_VER //ou _WIN32
#define _CRT_SECURE_NO_DEPRECATE //evita warning no MSVC
#endif

#include <stdio.h>
//#include <string.h>
#include <stdlib.h>
//#include <limits.h>
#include <math.h>
//#include <time.h>
#include <float.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>

#define MAX_LINE_SIZE 1024
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* dimensao do problema */
#define dim 2


/* min e a minima distancia encontrada */
/* cp1 e cp2 sao os pontos mais proximos */
double min = DBL_MAX;
int cp1[2];
int cp2[2];
int *middles;

/* numero de calculos de sqrt */
long long int comparacoes = 0;

/* Le o arquivo e pega seus valores */
int *parse(char* argv[], unsigned int* n);

/* Auxiliar para saber o tamanho do arquivo */
unsigned getSize(char* file);

/* lancamento de erros */
void error(char* s);
void usage(char* program);

void printV(int* P, unsigned int N);

/* Quicksort em x */

/* processo de merge do algoritmo em y e conquista */

/* Checa se a distancia em Y entre dois pontos eh menor que o min */
int check(int *p1, int *p2);
/* Verifica se os pontos estão em lados opostos em relação ao middle */
int checkX(int *p1, int *p2, int middle);
/* Checa se a distancia entre dois pontos é menor que o min */
void dist(int *p1, int *p2);
/* Se pass = 1 faz o mergesort em x */
/* Se pass = 2 faz o mergesort em y e resolve o problema de conquista do divide-conquer algorithm */
//Faz as verificações relacionados à distância delta
void delta_check(int *c, int middle, int **p1, int **p2, int **p3, int **p4);

void sortBU(int* arr, int size);
void mergeBU(int* src1, int* src2, int len1, int len2, int* dest, int middle);
int* determineMiddles(int *arr, int size);
void sortBU_X(int* arr, int size);
void mergeBU_X(int* src1, int* src2, int len1, int len2, int* dest);

int main(int argc, char* argv[]) {
	/* Temporizacao */
	double time = 0.0;
	double start, end;
	if (argc != 2)
		usage(argv[0]);

	//Parse
	start = omp_get_wtime();
	FILE* input;
	input = fopen(argv[1], "r");
	if (input == NULL)
		error("Erro na leitura do arquivo");
	unsigned int n;

	int* P = parse(argv, &n);

	//printV(P, n);

	fclose(input);
	end = omp_get_wtime();
	time = end - start;
	printf("Time parser: %.3f seconds\n", time);

	//OpenMP
	int num_threads;
#pragma omp parallel
	{
#pragma omp master
		{
			num_threads = omp_get_num_threads();
			printf("Numthreads: %d \n", num_threads);
		}
	}
	omp_set_nested(1);
	printf("Nested paralellism set to: %d\n", omp_get_nested());


	//Solucao
	start = omp_get_wtime();

	/* Ordenacao em X */
	sortBU_X(P, n);

	end = omp_get_wtime();
	time = end - start;
	printf("Time sort em X: %.3f seconds\n", time);
	//printV(P, n);
	/* Ordenacao em Y */
	sortBU(P, n);


	printf("Closest pair: [%d,%d] e [%d,%d] com dist: %.2f\n", cp1[0], cp1[1], cp2[0], cp2[1], min);
	printf("Comparacoes: %lld\n", comparacoes);

	free(P);

	end = omp_get_wtime();
	time = end - start;
	printf("Time solution: %.3f seconds\n", time);

	//printV(P, n); //se for ativar tem que comentar o free(P)

	//getchar();
	return 0;
}

void mergeBU(int* src1, int* src2, int len1, int len2, int* dest, int middle) {
	int idx1 = 0, idx2 = 0;
	int loc = 0;

	int *p1, *p2, *p3, *p4;
	p1 = NULL; p2 = NULL; p3 = NULL; p4 = NULL;

	//printf("middle recebido: %d\n", middle);

	while (idx1 < len1 && idx2 < len2) {
		if (src1[idx1*dim + 1] <= src2[idx2*dim + 1]) {
			dest[loc*dim + 0] = src1[idx1*dim + 0];
			dest[loc*dim + 1] = src1[idx1*dim + 1];
			delta_check(&dest[loc*dim], middle, &p1, &p2, &p3, &p4);
			idx1++;
		}
		else {
			dest[loc*dim + 0] = src2[idx2*dim + 0];
			dest[loc*dim + 1] = src2[idx2*dim + 1];
			delta_check(&dest[loc*dim], middle, &p1, &p2, &p3, &p4);
			idx2++;
		}
		loc++;
	}

	//copy the rest
	for (int i = idx1; i < len1; i++) {
		dest[loc*dim + 0] = src1[i*dim + 0];
		dest[loc*dim + 1] = src1[i*dim + 1];
		delta_check(&dest[loc*dim], middle, &p1, &p2, &p3, &p4);
		loc++;
	}

	for (int i = idx2; i < len2; i++) {
		dest[loc*dim + 0] = src2[i*dim + 0];
		dest[loc*dim + 1] = src2[i*dim + 1];
		delta_check(&dest[loc*dim], middle, &p1, &p2, &p3, &p4);
		loc++;
	}
}

void sortBU(int* arr, int size) {
	int *temp = (int*)malloc(sizeof(int*) * size * dim);
	//int temp[size];
	int *repo1, *repo2, *aux;
	int stIdx; //aqui para msvc nao reclamar

	repo1 = arr;
	repo2 = temp;

	int *middles = determineMiddles(arr, size);

	for (int grpSize = 1; grpSize < size; grpSize <<= 1) {
		//printf("gprSize = %d\n\n", grpSize);
#pragma omp parallel for
		for (stIdx = 0; stIdx < size; stIdx += 2 * grpSize) {
			int nextIdx = stIdx + grpSize;
			int secondGrpSize = MIN(MAX(0, size - nextIdx), grpSize);

			//para enchergar a uniao
			//printf("----------------\n");
			//printV(repo1 + stIdx*dim, MAX(0, MIN(grpSize, size - stIdx)));
			//printf("---\n");
			//printV(repo1 + nextIdx*dim, secondGrpSize);
			//printf("----------------\n");
			//if(secondGrpSize) printf("middle: %d\n\n\n", middles[stIdx + grpSize]);
			//else printf("\n\n\n");
			

			if (secondGrpSize == 0) {
				for (int i = 0; i < size - stIdx; i++) {
					repo2[(stIdx + i)*dim + 0] = repo1[(stIdx + i)*dim + 0];
					repo2[(stIdx + i)*dim + 1] = repo1[(stIdx + i)*dim + 1];
				}
			}
			else {
				mergeBU(repo1 + stIdx*dim, repo1 + nextIdx*dim, grpSize, secondGrpSize, repo2 + stIdx*dim, middles[stIdx + grpSize]);
			}
		}
		aux = repo1;
		repo1 = repo2;
		repo2 = aux;

	}

	//if (repo1 != arr) {
	//	memcpy(arr, temp, sizeof(int)*size*dim);
	//}

	free(temp);
}

int* determineMiddles(int *arr, int size) {
	int *middles = (int*)malloc(sizeof(int) * size);

	int i; //aqui para msvc nao reclamar

#pragma omp parallel for
	for (i = 0; i < size; i++) {
		middles[i] = arr[i*dim];
	}

	return middles;
}

void mergeBU_X(int* src1, int* src2, int len1, int len2, int* dest) {
	int idx1 = 0, idx2 = 0;
	int loc = 0;


	while (idx1 < len1 && idx2 < len2) {
		if (src1[idx1*dim] <= src2[idx2*dim]) {
			dest[loc*dim + 0] = src1[idx1*dim + 0];
			dest[loc*dim + 1] = src1[idx1*dim + 1];
			idx1++;
		}
		else {
			dest[loc*dim + 0] = src2[idx2*dim + 0];
			dest[loc*dim + 1] = src2[idx2*dim + 1];
			idx2++;
		}
		loc++;
	}

	//copy the rest
	for (int i = idx1; i < len1; i++) {
		dest[loc*dim + 0] = src1[i*dim + 0];
		dest[(loc++)*dim + 1] = src1[i*dim + 1];
	}

	for (int i = idx2; i < len2; i++) {
		dest[loc*dim + 0] = src2[i*dim + 0];
		dest[(loc++)*dim + 1] = src2[i*dim + 1];
	}
}

void sortBU_X(int* arr, int size) {
	int *temp = (int*)malloc(sizeof(int*) * size * dim);
	int *repo1, *repo2, *aux;
	int stIdx; //aqui para msvc nao reclamar

	repo1 = arr;
	repo2 = temp;

	for (int grpSize = 1; grpSize < size; grpSize <<= 1) {
#pragma omp parallel for
		for (stIdx = 0; stIdx < size; stIdx += 2 * grpSize) {
			int nextIdx = stIdx + grpSize;
			int secondGrpSize = MIN(MAX(0, size - nextIdx), grpSize);

			if (secondGrpSize == 0) {
				for (int i = 0; i < size - stIdx; i++) {
					repo2[(stIdx + i)*dim + 0] = repo1[(stIdx + i)*dim + 0];
					repo2[(stIdx + i)*dim + 1] = repo1[(stIdx + i)*dim + 1];
				}
			}
			else {
				mergeBU_X(repo1 + stIdx*dim, repo1 + nextIdx*dim, grpSize, secondGrpSize, repo2 + stIdx*dim);
			}
		}
		aux = repo1;
		repo1 = repo2;
		repo2 = aux;

	}

	if (repo1 != arr) {
		memcpy(arr, temp, sizeof(int)*size*dim);
	}

	free(temp);
}

/*
Verifica se os pontos existem e a distancia em Y entre eles e menor que min
Return 1 : sim
Return 0 : nao
*/
int check(int *p1, int *p2)
{
	if ((p1 != NULL) && (p2 != NULL) && ((double)(p1[1] - p2[1]) < min)) return 1;
	return 0;
}

/*
Verifica a posição em X dos pontos em referência ao middle
Return 1 : Lados opostos
Return 0 : Mesmos lados
*/
int checkX(int *p1, int *p2, int middle)
{
	if ((p1[0] - middle)*(p2[0] - middle) <= 0) return 1;
	return 0;
}

/*
Algoritmo que calcula a distancia entre dois pontos, e se for menor que delta, substituimos por esses novos valores.
Return 1 : distancia calculada foi menor que delta
Return 0 : distancia calculada foi maior que delta
*/
void dist(int *p1, int *p2)
{
	double dist;
	dist = sqrt((double)(p1[0] - p2[0])*(p1[0] - p2[0]) + (double)(p1[1] - p2[1])*(p1[1] - p2[1]));
	comparacoes++;
#pragma omp critical (distCrit)
	{
		if (dist < min) {
			min = dist;
			cp1[0] = p1[0]; cp1[1] = p1[1];
			cp2[0] = p2[0]; cp2[1] = p2[1];
		}
	}

}


/*
Algoritmo de calculo do closest pair
*/

void delta_check(int *c, int middle, int **p1, int **p2, int **p3, int **p4)
{
	if (fabs(c[0] - middle) < min) { //min aqui nao precisa ser critico
		if (check(&c[0], *p4)) {

			if (checkX(&c[0], *p4, middle)) dist(&c[0], *p4); else;

			if (check(&c[0], *p3)) {

				if (checkX(&c[0], *p3, middle)) dist(&c[0], *p3); else;

				if (check(&c[0], *p2)) {

					if (checkX(&c[0], *p2, middle)) dist(&c[0], *p2); else;

					if (check(&c[0], *p1)) {
						if (checkX(&c[0], *p1, middle)) dist(&c[0], *p1); else;
					}
					*p1 = *p2; *p2 = *p3; *p3 = *p4; *p4 = &c[0];
				}
				else {
					*p1 = NULL; *p2 = *p3; *p3 = *p4; *p4 = &c[0];
				}
			}
			else {
				*p1 = NULL; *p2 = NULL; *p3 = *p4; *p4 = &c[0];
			}
		}
		else {
			*p1 = NULL; *p2 = NULL; *p3 = NULL; *p4 = &c[0];
		}
	}
}

int *parse(char* argv[], unsigned int* n) {
	printf("File: '%s'\n", argv[1]);
	FILE* input;
	input = fopen(argv[1], "rb");
	if (input == NULL) {
		printf("Erro na leitura do arquivo de entrada.");
		return 0;
	}

	unsigned int filesize = getSize(argv[1]);

	printf("Dim: %d, Points: %u\n", dim, (unsigned)(filesize / sizeof(int)) / dim);

	int *P = (int*)malloc(filesize);
	if (P == NULL) error("Memory Allocation Failed");

	fread(P, sizeof(int), filesize / sizeof(int), input);

	fclose(input);

	*n = (filesize / sizeof(int)) / dim;
	return P;
}

/*
Retorna o tamanho do arquivo em bytes de 'file'
*/

unsigned getSize(char* file) {
	struct _stat buf;
	unsigned result = _stat(file, &buf);
	if (result != 0)
	{
		printf("Erro na leitura do tamanho do arquivo.");
		return 0;
	}
	unsigned long tam = buf.st_size;

	return tam;
}

void usage(char* program)
{
	printf("usage: %s FILENAME\n", program);
	exit(1);
}

void error(char* s)
{
	perror(s);
	exit(1);
}

void printV(int* P, unsigned int N) {
	//printf("debug:\n");
	for (unsigned int i = 0; i < N; i++) {
		printf("[%10d], [%10d]", P[i*dim + 0], P[i*dim + 1]);
		printf("\n");
	}
}