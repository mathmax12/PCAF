/*
 * common.h
 *
 *  Created on: Dec 11, 2013
 *      Author: ryan
 */

#ifndef COMMON_H_
#define COMMON_H_

//#define EST

#include <sys/time.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <queue>
using namespace std;

#include <algorithm>
#include <string>
#include <map>
#include <cassert>
#include <limits>
#include <cmath>

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/algorithms/dist.h"
#include "flann/util/random.h"
#include "flann/algorithms/center_chooser.h"
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/allocator.h"
#include "flann/util/saving.h"
#include "flann/util/logger.h"
#include "flann/util/lsh_table.h"

using namespace flann;

#define TIMER_T                         struct timeval
#define TIMER_READ(time)                gettimeofday(&(time), NULL)
#define TIMER_DIFF_SECONDS(start, stop) \
    (((double)(stop.tv_sec)  + (double)(stop.tv_usec / 1000000.0)) - \
     ((double)(start.tv_sec) + (double)(start.tv_usec / 1000000.0)))

inline uint64_t ticks() {
	uint32_t tmp[2];
	__asm__ ("rdtsc" : "=a" (tmp[1]), "=d" (tmp[0]) : "c" (0x10) );
	return (((uint64_t) tmp[0]) << 32) | ((uint64_t) tmp[1]);
}

extern "C" {
	float vdistance_(const float*, const float*, size_t);
}

#endif /* COMMON_H_ */
