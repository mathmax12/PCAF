#include "common.h"
#include "flann/algorithms/dist.h"
using namespace flann;

#define ED 1

#if ED == 0

L2<float> distance_;
float vdistance_(const float* v1, const float* v2, size_t len) {
	return distance_(v1, v2, len);
}

#elif ED == 1

float vdistance_(const float* v1, const float* v2, size_t len) {
	float dist = 0;
	#pragma novector
	for (size_t i = 0; i < len; ++i) {
		float tmp = v1[i] - v2[i];
		dist += tmp * tmp;
	}
	return dist;
}

#elif ED == 2

float vdistance_(const float* v1, const float* v2, size_t len) {
	float dist, d0, d1;
	float *vec1 = (float*)v1, *vec2 = (float*)v2;
	int t = len / 2;
	dist = 0;
	//#pragma novector
	for(int i = 0; i < t; ++i){
		d0 = vec1[0] - vec2[0];
		d1 = vec1[1] - vec2[1];
		d0 = d0 * d0;
		d1 = d1 * d1;
		dist += d0 + d1;
		vec1 += 2;
		vec2 += 2;
	}

	t = len % 2;
	if (t > 0) {
		d0 = vec1[0] - vec2[0];
		dist += d0 * d0;
	}

	return dist;
}

#elif ED == 4

float vdistance_(const float* v1, const float* v2, size_t len) {
	float dist, d0, d1, d2, d3;
	float *vec1 = (float*)v1, *vec2 = (float*)v2;
	int t = len / 4;
	dist = 0;
	//#pragma novector
	for(int i = 0; i < t; ++i){
		d0 = vec1[0] - vec2[0];
		d1 = vec1[1] - vec2[1];
		d2 = vec1[2] - vec2[2];
		d3 = vec1[3] - vec2[3];
		d0 = d0 * d0;
		d1 = d1 * d1;
		d2 = d2 * d2;
		d3 = d3 * d3;
		dist += d0 + d1 + d2 + d3;
		vec1 += 4;
		vec2 += 4;
	}

	t = len % 4;
	#pragma novector
	for(int i = 0; i < t; ++i){
		d0 = vec1[i] - vec2[i];
		dist += d0 * d0;
	}

	return dist;
}

#elif ED == 5

#include <immintrin.h>
float vdistance_(const float* v11, const float* v22, size_t len) {
        float dist = 0;
#ifdef __MIC__
        __m512 _v1, _v2, _dist;
        int t1 = len / 16;
        int t2 = len % 16;
        float *v1 = (float*)v11, *v2 = (float*)v22;
        _dist = _mm512_setzero_ps();
        for(int i = 0; i < t1; ++i){
                _v1 = _mm512_loadunpacklo_ps(_v1, v1);
                _v1 = _mm512_loadunpackhi_ps(_v1, v1 + 16);
                _v2 = _mm512_loadunpacklo_ps(_v2, v2);
                _v2 = _mm512_loadunpackhi_ps(_v2, v2 + 16);
                _v1 = _mm512_sub_ps(_v1, _v2);
                _dist = _mm512_fmadd_ps(_v1, _v1, _dist);
                v1 += 16;
                v2 += 16;
        }

        dist = _mm512_reduce_add_ps(_dist);
        #pragma novector
        for(int i = 0; i < t2; ++i) {
                float tmp = v1[i] - v2[i];
                dist += tmp * tmp;
        }
#else
      //__m256 _a, _b, _c;
      //      int t1 = len / 8;
      //      int t2 = len % 8;
#endif

	return dist;
}

#endif
