
#include     "stdio.h"
#include     "ipp.h"
#include     "intrin.h"
#include     <cstdlib>
#include	 <ctime>

#define        VECTOR_SIZE    10000

int main() {

	Ipp64u start, end;
	Ipp64u time1, time2;

	float *in = new  float[VECTOR_SIZE];


	//initialize
	for (int i = 0; i < VECTOR_SIZE; i++)
		in[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 100.0));

	//serial
	start = ippGetCpuClocks();

	float max = 0;

	for (int i = 0; i < VECTOR_SIZE; i++)
		max = (max < in[i]) ? in[i] : max;

	end = ippGetCpuClocks();
	time1 = end - start;
	printf("\nmax of value with serial way= %f", max);
	printf("\nTime for Serial = %d \n", (Ipp32s)time1);

	//parallel -------------------------------

	start = ippGetCpuClocks();

	__m128 vec;
	__m128 Max = _mm_setzero_ps();

	for (int i = 0; i < VECTOR_SIZE; i += 4) {
		vec = _mm_loadu_ps(&in[i]);
		Max = _mm_max_ps(Max, vec);
	}

	__m128 sh1 = _mm_shuffle_ps(Max, Max, _MM_SHUFFLE(3, 3, 2, 3));
	__m128 maxsh1 = _mm_max_ps(Max, sh1);
	__m128 sh2 = _mm_shuffle_ps(maxsh1, maxsh1, _MM_SHUFFLE(3, 3, 3, 3));
	__m128 maxsh2 = _mm_max_ps(maxsh1, sh2);

	max = _mm_cvtss_f32(maxsh2);

	end = ippGetCpuClocks();
	delete[]in;
	time2 = end - start;
	printf("\nmax of value with Parallel way= %f", max);
	printf("\nTime for Parallel = %d \n", (Ipp32s)time2);
	printf("\nSpeedup = %f\n", (float)(time1) / (float)time2);

	return 0;
}