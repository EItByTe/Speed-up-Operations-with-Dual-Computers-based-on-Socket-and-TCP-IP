#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <math.h>
#include <Windows.h>
#include "device_functions.h"
#include <omp.h>
#define DATA_SIZE 128000000
#define THREAD_NUM 256
// Block数量
#define BLOCK_NUM 32
#define Row 8000
#define Col 8000       //定义计算的二维数组结构为8000*8000，为总数据的一半
#define MAX_ONCE(a,b) (((a) > (b)) ? (a):(b))
float data[DATA_SIZE];
int clockRate;

/* 产生随机数 */
void generateNumbers(float* numbers, int size) {
    int i;
    for (i = 0; i < size; i++) {
        numbers[i] = float(rand() + rand() + rand() + rand());
    }
}

/* 打印GPU设备信息 */
void printDeviceProps(const cudaDeviceProp* prop) {
    printf("Device Name: %s\n", prop->name);
    printf("totalGlobalMem: %ld\n", prop->totalGlobalMem);
    printf("sharedMemPerBlock: %d\n", prop->sharedMemPerBlock);
    printf("regsPerBlock: %d\n", prop->regsPerBlock);
    printf("warpSize: %d\n", prop->warpSize);
    printf("memPitch: %d\n", prop->memPitch);
    printf("maxThreadPerBlock: %d\n", prop->maxThreadsPerBlock);
    printf("maxThreadsDim[0-2]: %d %d %d\n", prop->maxThreadsDim[0], prop->maxThreadsDim[1], prop->maxThreadsDim[2]);
    printf("maxGridSize[0-2]: %d %d %d\n", prop->maxGridSize[0], prop->maxGridSize[1], prop->maxGridSize[2]);
    printf("totalConstMem: %d\n", prop->totalConstMem);
    printf("major: %d & minor: %d\n", prop->major, prop->minor);
    printf("clockRate: %d\n", prop->clockRate); clockRate = prop->clockRate;
    printf("textureAlignment: %d\n", prop->textureAlignment);
    printf("deviceOverlap: %d\n", prop->deviceOverlap);
    printf("multiProcessorCount: %d\n", prop->multiProcessorCount);
}

/* CUDA 初始化 */
bool initCUDA() {
    int count, i;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&count);

    if (0 == count) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    for (i = 0; i < count; i++) {
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    cudaSetDevice(i);

    //printDeviceProps(&prop);

    return true;
}

float pure_max(const float data[], const int len)
{
    double max_temp = 0;
    for (int i = 0; i < len; i++) {
        if (log(sqrt(data[i])) > max_temp)
            max_temp = log(sqrt(data[i]));
    }
    return float(max_temp);
}
/* 计算最大耗时 */
clock_t findMaxTimeUsed(const clock_t* time) {
    int i;
    clock_t min_start = time[0], max_end = time[BLOCK_NUM];
    for (i = 0; i < BLOCK_NUM; i++) {
        if (time[i] < min_start) {
            min_start = time[i];
        }
        if (time[i + BLOCK_NUM] > max_end) {
            max_end = time[i + BLOCK_NUM];
        }
    }
    return max_end - min_start;
}

/* 计算和（__global__函数运行于GPU）*/
__global__ static void sumOfSquares(float* numbers, float* sub_sum, clock_t* time) {
    int i;

    // 获取当前线程所属的Block号（从0开始）
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    if (thread_id == 0) {
        time[block_id] = clock();
    }

    sub_sum[block_id * THREAD_NUM + thread_id] = 0;
    // Block0-线程0获取第0个元素，Block0-线程1获取第1个元素...Block1-线程0获取第THREAD_NUM个元素，以此类推... 
    for (i = block_id * THREAD_NUM + thread_id; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
        sub_sum[block_id * THREAD_NUM + thread_id] += log10f(sqrtf(numbers[i]));
    }

    if (thread_id == 0) {
        time[block_id + BLOCK_NUM] = clock();
    }
}

/* 计算和（__global__函数运行于GPU）*/
__global__ static void cal_max(float* numbers, float* sub_sum, clock_t* time) {
    int i;

    // 获取当前线程所属的Block号（从0开始）
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    if (thread_id == 0) {
        time[block_id] = clock();
    }

    sub_sum[block_id * THREAD_NUM + thread_id] = 0;
    // Block0-线程0获取第0个元素，Block0-线程1获取第1个元素...Block1-线程0获取第THREAD_NUM个元素，以此类推... 
    for (i = block_id * THREAD_NUM + thread_id; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
        sub_sum[block_id * THREAD_NUM + thread_id] += log10f(sqrtf(numbers[i]));
    }

    if (thread_id == 0) {
        time[block_id + BLOCK_NUM] = clock();
    }
}

__global__ void cudacalculate(float** C, float** A)
//定义cuda加速计算函数cudacalculate
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    //定义线程的线程编号
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < Col && idy < Row) {
        C[idy][idx] = log(sqrt(A[idy][idx]));
        //定义矩阵对位求开平方取log的指定运算操作
    }
}

__global__ void maxCuda(float d_a[DATA_SIZE], float d_a_temp[DATA_SIZE], float dat_Max[BLOCK_NUM])
{
    __shared__ float Max_temp[THREAD_NUM];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
     int tid = threadIdx.x;
    d_a_temp[i] = log(sqrt(d_a[i]));
    Max_temp[tid] = d_a_temp[tid + blockIdx.x * blockDim.x];
    __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            float temp1 = Max_temp[tid + stride];
            if (Max_temp[tid] < temp1) {
                float temp = Max_temp[tid];
                Max_temp[tid] = temp1;
                Max_temp[tid + stride] = temp;
                }
        }
    }
        if (tid == 0)
            dat_Max[blockIdx.x] = Max_temp[0];
}


int main(void) {
    if (!initCUDA()) {
        return 0;
    }
    float* gpudata;
    int i;
    double sum;
    float sub_sum[BLOCK_NUM * THREAD_NUM], * gpu_sub_sum;
    // 每个Block设置一个计时单元
    clock_t time_used[BLOCK_NUM * 2], * gpu_time_used, start, finish;
    float cpucosttime;
    generateNumbers(data, DATA_SIZE);

    /*------------------------------------------------*/
    float* Mat_d;
    float* Mat_d_temp;
    float* dat_Max;
    float ret_Max[BLOCK_NUM];
    cudaMalloc((void**)&Mat_d, DATA_SIZE * sizeof(float));
    cudaMalloc((void**)&Mat_d_temp, DATA_SIZE * sizeof(float));
    cudaMalloc((void**)&dat_Max, DATA_SIZE * sizeof(float));
    start = clock();
    cudaMemcpy(Mat_d, data, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);//把数据从Host 传到 Device
    maxCuda << <BLOCK_NUM, THREAD_NUM >> > (Mat_d, Mat_d_temp, dat_Max); //调用内核函数
    cudaMemcpy(ret_Max, dat_Max, sizeof(float) * BLOCK_NUM, cudaMemcpyDeviceToHost);//将结果传回到主机端
    //使用 cpu 多线程取返回数组中的最大值
    float max = 0;
    float maxx[BLOCK_NUM] = {0};
    for (int i = 0; i < BLOCK_NUM; i++) {
        maxx[i] = MAX_ONCE(maxx[i], ret_Max[i]);
    }
    for (int i = 0; i < BLOCK_NUM; i++)
        max = MAX_ONCE(maxx[i], max);
    finish = clock();
    double gpumaxcosttime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("GPU max is: %f, time used : %lf(s)\n", max, gpumaxcosttime);
    //释放显存空间
    cudaFree(Mat_d);
    cudaFree(Mat_d_temp);
    cudaFree(dat_Max);

    float max_cpu;
    /*--------------无加速求最大值开始--------------*/
    start = clock();
    max_cpu = pure_max(data, DATA_SIZE);
    finish = clock();
    float cpumaxcosttime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("CPU max is: %f, time used : %f(s)\n", max_cpu, cpumaxcosttime);
    /*--------------无加速求最大值结束--------------*/
    /*------------------------------------------------*/

    /*求和部分*/
    cudaMalloc((void**)&gpudata, sizeof(float) * DATA_SIZE);
    // 当前一共有BLOCK_NUM * THREAD_NUM个线程
    cudaMalloc((void**)&gpu_sub_sum, sizeof(float) * BLOCK_NUM * THREAD_NUM);
    cudaMalloc((void**)&gpu_time_used, sizeof(clock_t) * BLOCK_NUM * 2);

    cudaMemcpy(gpudata, data, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice);
    // 更新Block数
    start = clock();
    sumOfSquares << < BLOCK_NUM, THREAD_NUM, 0 >> > (gpudata, gpu_sub_sum, gpu_time_used);

    cudaMemcpy(time_used, gpu_time_used, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(sub_sum, gpu_sub_sum, sizeof(float) * BLOCK_NUM * THREAD_NUM, cudaMemcpyDeviceToHost);

    sum = 0.0f;

    for (i = 0; i < BLOCK_NUM * THREAD_NUM; i++) {
        sum += sub_sum[i];
    }
    finish = clock();
    float gpucosttime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("GPU sum is: %f, time used : %f(s)\n", sum, gpucosttime);
    cudaFree(gpudata);
    cudaFree(gpu_sub_sum);
    cudaFree(time);

    //cpu
    start = clock();
    sum = 0.0f;
    LARGE_INTEGER  start2 = { 0 }; LARGE_INTEGER  end2 = { 0 };
    for (i = 0; i < DATA_SIZE; i++) {
        sum += log10f(sqrtf(data[i]));
    }
    finish = clock();
    cpucosttime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("CPU sum is: %f, time used : %f(s)\n", sum, cpucosttime);

    system("pause");

    return 0;
}