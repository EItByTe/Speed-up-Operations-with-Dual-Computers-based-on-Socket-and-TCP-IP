#define _WINSOCK_DEPRECATED_NO_WARNINGS
/*陈晨 1852731 自动化*/
/*张儒戈 1951393 自动化*/
/*-----socket用到-----*/
#pragma comment(lib,"ws2_32.lib")
#include <WinSock2.h>
#include <stdlib.h>
#include <time.h>
/*------------------*/
#include <iostream>
#include <math.h>
#include <omp.h>		//add for omp
#include <Windows.h>	//add for taking frequence
#include <immintrin.h>  

using namespace std;

//#define NOSPEEDUP
#define FINALSPEEDUP
#define COMMUNICATION

#define INET_ADDR "192.168.200.2"//更换网络需要更改，zrg的手机热点
#define MAX_THREADS 64
//双机时总的数据量2000000，所以单机1000000
//#define SUBDATANUM 2000000
#define SUBDATANUM 1000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*这个数值是总数据量*/

//最快的两个数取最大值，稍快一些
#define MAX_ONCE(a,b) (((a) > (b)) ? (a):(b))		//max()

//待测试数据定义为：
float rawFloatData[DATANUM];
float rawIntData[DATANUM];
unsigned char trans_sort[4*DATANUM];
//float rawFloatData3[DATANUM];
//排序的结果也放在全局变量中，main中放不了这么大的
float sort_nosp[DATANUM];
/*----------------函数声明----------------*/
//data是原始数据，len为长度。结果通过函数返回
float pure_sum(const float data[], const int len); 
//data是原始数据，len为长度。结果通过函数返回
float pure_max(const float data[], const int len);
//data是原始数据，start是起始位置,end是终止位置。排序结果在result中。
void pure_sort(const float data[], const int start, const int end, float  result[]);
/*判断排序是否成功,如果失败打印false，并且返回-2，如果成功打印true，并且返回0*/
int sort_result_check(float* array, int len);
float omp_sum(const float data[], const int len); //data是原始数据，len为长度。结果通过函数返回
float omp_max(const float data[], const int len);//data是原始数据，len为长度。结果通过函数返回
void omp_sort(const int end, float data[], const int len);//data是原始数据，len为长度。排序结果直接体现在RawFloatData中
float avx_sum(float data[], int len);	//data是原始数据，len为长度，返回值为总和
float avx_max(float data[], int len);
/*-----------------------------------------*/

/*----------------无加速算法----------------*/
float pure_sum(const float data[], const int len)
{
	double sum = 0.0f;
	for (int i = 0; i < len; i++)
		sum += log(sqrt(data[i]));
	return float(sum);
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
//采用归并排序 （从小到大）
void pure_sort(float data[], const int start, const int end, float result[])
{
	if (end - start > 1) {
		int m = start + (end - start) / 2;
		int p = start, q = m, i = start;
		pure_sort(data, start, m, result);
		pure_sort(data, m, end, result);

		while (p < m || q < end) {
			if (q >= end || (p < m && log(sqrt(data[p])) <= log(sqrt(data[q])))) {	//在这里加log(sqrt())增加一些计算时间
				result[i++] = data[p++];
			}
			else {
				result[i++] = data[q++];
			}
		}
		for (i = start; i < end; i++) {
			data[i] = result[i];
		}
	}
}
/*----------------------------------------*/
/*----------------OpenMP加速----------------*/
float omp_sum(const float data[], const int len)
{
	double sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < len; i++)
		sum += log(sqrt(data[i]));
	return float(sum);
}

float omp_max(const float data[],const int len) //omp求最大值
{
	double max_omp = 0.0;
	float max_temp[MAX_THREADS] = { 0.0 };
#pragma omp parallel for 
	for (int thd = 0; thd < MAX_THREADS; thd++) {

		//printf("i = %d, I am Thread %d\n", thd, omp_get_thread_num());

		for (int i = 0; i < SUBDATANUM; i++) {

			max_temp[thd] = MAX_ONCE(log(sqrt(data[i + MAX_THREADS * thd])), max_temp[thd]);
		}
	}

	for (int i = 0; i < MAX_THREADS; i++) {
		max_omp = MAX_ONCE(max_omp, max_temp[i]);
	}

	return float(max_omp);
}

//合并两个区间
void merge(const int l1, const int r1, const int r2, float data[], float temp[]) {
	int top = l1, p = l1, q = r1;
	while (p < r1 || q < r2) {
		if (q >= r2 || (p < r1 && log(sqrt(data[p])) <= log(sqrt(data[q])))) {	//在这里加log(sqrt())增加一些计算时间
			temp[top++] = data[p++];
		}
		else {
			temp[top++] = data[q++];
		}
	}
	for (top = l1; top < r2; top++) {
		data[top] = temp[top];
	}
}
void omp_sort(const int end, float data[], const int len) {
	int i, j;
	float t;
	float* temp;
	temp = (float*)malloc(len * sizeof(float));
//这里做了一些优化，预处理合并了单个的区间，略微提高的速度
#pragma omp parallel for private(i, t) shared(len, data)
	for (i = 0; i < len / 2; i++)
		if (log(sqrt(data[i * 2])) > log(sqrt(data[i * 2 + 1]))) {
			t = data[i * 2];
			data[i * 2] = data[i * 2 + 1];
			data[i * 2 + 1] = t;
		}
//i代表每次归并的区间长度，j代表需要归并的两个区间中最小的下标
	for (i = 2; i < end; i *= 2) {
#pragma omp parallel for private(j) shared(end, i)
		for (j = 0; j < end - i; j += i * 2) {
			merge(j, j + i, (j + i * 2 < end ? j + i * 2 : end), data, temp);
		}
	}
}
/*------------------------------------------*/
/*------------------AVX加速------------------*/
float avx_sum(float data[], int len) //用AVX加速求和
{
	__m256* ptr = (__m256*)data; //256bit数据类型，8个并行浮点数构成  _m256==256位紧缩单精度（AVX）
	__m256 xfsSum = _mm256_setzero_ps();	//Sets float32 YMM registers to zero
	double s = 0;	//返回值
	const float* q;	//指针传值用
	for (int i = 0; i < len / 8; ++i, ++ptr)
	{
		//__m256 sqr = _mm256_sqrt_ps(*ptr);	//先取平方根，增加计算时间
		//__m256 lgy = _mm256_log_ps(sqr);	//再做log运算，增加计算时间
		xfsSum = _mm256_add_ps(xfsSum, _mm256_log_ps(_mm256_sqrt_ps(*ptr)));//求和
	}
	q = (const float*)&xfsSum;
	s = (q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7]);
	return float(s);
}

float avx_max(float data[], int len)
{
	float* result;
	float max_last = 0.0f;
	__m256* ptr = (__m256*)data; //强制转换成m256数据类型，由8个并行数据类型构成
	__m256 max_temp = _mm256_setzero_ps();
//#pragma omp parallel for
	for (int i = 0; i < len / 8; ++i)
	{
		__m256 lgy = _mm256_log_ps(_mm256_sqrt_ps(*ptr));
		max_temp = _mm256_max_ps(max_temp, lgy);	//取最大值
		++ptr;
	}
	result = (float*)&max_temp;
	//再对这8个值进行求个最大
	for (int i = 0; i < 8; i++)
	{
		max_last = MAX_ONCE(result[i], max_last);
	}
	return max_last;
}
/*------------------------------------------*/
/*------------------AVX、OMP加速------------------*/
float avx_omp_sum(float data[], int len) //用avx加速求和
{
	__m256* ptr = (__m256*)data; //256bit数据类型，8个并行浮点数构成  _m256==256位紧缩单精度（AVX）
	//__m256 xfsSum = _mm256_setzero_ps();	//Sets float32 YMM registers to zero
	double sum = 0;	//返回值
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < len / 8; i++)
	{
		//__m256 sqr = _mm256_sqrt_ps(*ptr);	//先取平方根，增加计算时间
		__m256 xfsSum = _mm256_log_ps(_mm256_sqrt_ps(_mm256_loadu_ps(data)));	//再做log运算，增加计算时间
		///xfsSum = _mm256_add_ps(xfsSum, lgy);//求和
		data += 8;
		float *q = (float*)&xfsSum;
		sum += (q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7]);
	}
	return float(sum);
}

float avx_omp_max(float data[], int len)
{
	double max_last = 0.0f;
	double max_temp_avx[MAX_THREADS] = {0};
#pragma omp parallel for
	for (int i = 0; i < len / 8; i++)
	{
		__m256 Xfsmax = _mm256_log_ps(_mm256_sqrt_ps(_mm256_loadu_ps(data)));
		float* q = (float*)&Xfsmax;
		int id = omp_get_thread_num();
		for (int j = 0; j < 8; j++) {
			max_temp_avx[id] = MAX_ONCE(max_temp_avx[id], q[j]);
		}
	}
	//再对这8个值进行求个最大
	for (int i = 0; i < MAX_THREADS; i++){
		max_last = MAX_ONCE(max_temp_avx[i], max_last);
	}
	return float(max_last);
}
/*判断排序是否成功,如果失败打印false，并且返回-2，如果成功打印true，并且返回0*/
int sort_result_check(float* array, int len) //判断排序是否成功
{
	for (int j = 0; j < len - 1; j++)
	{
		if (array[j] > array[j + 1])
		{
			std::cout << "错误的" << endl;
			return -2;
		}
	}
	std::cout << "正确的" << endl;
	return 0;
}

/*浮点数组转字符数组，发送是以字节为单位，若不转换，会出错，len为要转换的数据长度*/
void floatArr2charArr(float* floatArr, unsigned int len, unsigned char* charArr) {
	unsigned int position = 0;
	unsigned char* temp = nullptr;
	for (int i = 0; i < len; i++) {
		temp = (unsigned char*)(&floatArr[i]);
		for (int k = 0; k < 4; k++) {
			charArr[position++] = *temp++;
		}
	}
}
/*测试用*/
//void intArr2charArr(int* intArr, unsigned int len, unsigned char* charArr) {
//	unsigned int position = 0;
//	unsigned char* temp = nullptr;
//	for (int i = 0; i < len; i++) {
//		temp = (unsigned char*)(&intArr[i]);
//		for (int k = 0; k < 4; k++) {
//			charArr[position++] = *temp++;
//		}
//	}
//}

int main()
{    
#ifdef COMMUNICATION
	/*-------------------------通信初始化------------------------*/
	WSAData wsaData;
	WORD DllVersion = MAKEWORD(2, 1);
	if (WSAStartup(DllVersion, &wsaData) != 0)
	{
		MessageBoxA(NULL, "WinSock startup error", "Error", MB_OK | MB_ICONERROR);
		exit(1);
	}
	SOCKADDR_IN addr;
	int sizeofaddr = sizeof(addr);//Adres przypisany do socketu Connection
	addr.sin_addr.s_addr = inet_addr(INET_ADDR); //localhost
	addr.sin_port = htons(1111); //target Port 服务器端必须绑定端口号，客户端不需要绑定端口号，客户端要告诉服务器自己的端口号，才能得到服务器的回复
	addr.sin_family = AF_INET; //IPv4 Socket

	SOCKET Connection = socket(AF_INET, SOCK_STREAM, NULL); //stream就有流速和流量
	if (connect(Connection, (SOCKADDR*)&addr, sizeofaddr) != 0) //Connection
	{
		MessageBoxA(NULL, "Bad Connection", "Error", MB_OK | MB_ICONERROR);
		return 0;
	}
	srand(time(NULL));
#endif
	/*----------------------------------------------------------*/
	LARGE_INTEGER m_liPerfFreq = { 0 };
	QueryPerformanceFrequency(&m_liPerfFreq);//获取频率
	//数据初始化
	for (size_t i = 0; i < DATANUM; i++)//数据初始化
	{
		//rawFloatData[i] = float(i + 1);//这样初始就始终有序了
		rawFloatData[i] = float((rand() + rand()));//类似于打乱，但是seed一直不变，所以答案每次打开sln时都是唯一的
	}
	float sum_nosp, sum_sp_openmp, sum_sp_avx;
	float max_nosp, max_sp_openmp, max_sp_avx;

#ifdef COMMUNICATION
	char start_flag = 2;
	/*从机端初始化完毕，发送标志位，让服务器端开始计算、计时*/
	cout << "从机请求主机协同计算 " << endl;
	int bytes_length = send(Connection, (char*)&start_flag, 1, NULL);//flag
#endif
	//LARGE_INTEGER  start = { 0 }; LARGE_INTEGER  end = { 0 };
	//QueryPerformanceCounter(&start);
	///*-----------------------------无加速版本------------------------------*/
#ifdef NOSPEEDUP
	/*----------------无加速求和开始---------------*/
	LARGE_INTEGER  start1 = { 0 }; LARGE_INTEGER  end1 = { 0 };
	QueryPerformanceCounter(&start1);
	sum_nosp = pure_sum(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end1);
	std::cout << "无加速求和 Time Consumed:" << ((end1.QuadPart - start1.QuadPart) * 1000/ m_liPerfFreq.QuadPart) <<"ms"<< endl;
	std::cout << "无加速求和结果为:" << sum_nosp << endl;
	/*----------------无加速求和结束---------------*/

	/*--------------无加速求最大值开始--------------*/
	LARGE_INTEGER  start2 = { 0 }; LARGE_INTEGER  end2 = { 0 };
	QueryPerformanceCounter(&start2);
	max_nosp = pure_max(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end2);
	std::cout << "无加速求最大值 Time Consumed:" << ((end2.QuadPart - start2.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	std::cout << "无加速求最大值结果为:" << max_nosp << endl;
	/*--------------无加速求最大值结束--------------*/

	/*----------------无加速排序开始----------------*/
	LARGE_INTEGER  start3 = { 0 }; LARGE_INTEGER  end3 = { 0 };
	QueryPerformanceCounter(&start3);
	pure_sort(rawFloatData, 0, DATANUM, sort_nosp);
	QueryPerformanceCounter(&end3);
	////也可以通过以下代码检查正确性，但是会刷屏，拖慢
	/*
	for (int a = 0; a < DATANUM; a++)
		cout << rawFloatData[a] << " ";
	*/
	std::cout << "无加速排序 Time Consumed:" << ((end3.QuadPart - start3.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	std::cout << "无加速排序是:";
	sort_result_check(sort_nosp, DATANUM);
	///*----------------无加速排序结束----------------*/
#endif
	///*---------------------------------------------------------------------*/
	/*-----------------------------OpenMP加速------------------------------*/

	/*----------------OMP加速求和开始---------------*/
	//LARGE_INTEGER  start4 = { 0 }; LARGE_INTEGER  end4 = { 0 };
	//QueryPerformanceCounter(&start4);
	//sum_sp_openmp = omp_sum(rawFloatData, DATANUM);
	//QueryPerformanceCounter(&end4);
	//std::cout << "OMP加速求和 Time Consumed:" << ((end4.QuadPart - start4.QuadPart)  * 1000/ m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "OMP加速求和结果为:" << sum_sp_openmp << endl;
	/*----------------OMP加速求和结束---------------*/

	///*--------------OMP加速求最大值开始--------------*/
	//LARGE_INTEGER start5 = { 0 }; LARGE_INTEGER end5 = { 0 };
	//QueryPerformanceCounter(&start5);
	//max_sp_openmp = omp_max(rawFloatData, DATANUM);
	//QueryPerformanceCounter(&end5);
	//std::cout << "omp加速求最大值 Time Consumed:" << ((end5.QuadPart - start5.QuadPart) * 1000/ m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "omp加速求最大值结果为:" << max_sp_openmp << endl;
	///*--------------OMP加速求最大值结束--------------*/

	/*---------------------------------------------------------------------*/

	/*-----------------------------AVX加速------------------------------*/
#ifdef FINALSPEEDUP
	/*----------------AVX加速求和开始----------------*/
	//LARGE_INTEGER  start7 = { 0 }; LARGE_INTEGER  end7 = { 0 };
	//QueryPerformanceCounter(&start7);
	sum_sp_avx = avx_sum(rawFloatData, DATANUM);
	//sum_sp_avx = avx_omp_sum(rawFloatData, DATANUM);	//avx+omp
	//QueryPerformanceCounter(&end7);    
	//std::cout << "AVX加速求和 Time Consumed:" << ((end7.QuadPart - start7.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "AVX加速求和结果为:" << sum_sp_avx << endl;
	/*----------------AVX加速求和结束----------------*/

	/*--------------AVX加速求最大值开始--------------*/
	//LARGE_INTEGER start8 = { 0 }; LARGE_INTEGER end8 = { 0 };
	//QueryPerformanceCounter(&start8);
	max_sp_avx = avx_max(rawFloatData, DATANUM);
	//max_sp_avx = avx_omp_max(rawFloatData, DATANUM); //avx+omp
	//QueryPerformanceCounter(&end8);
	//std::cout << "AVX加速求最大值 Time Consumed:" << ((end8.QuadPart - start8.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "AVX加速求最大值结果为:" << max_sp_avx << endl;
	/*--------------AVX加速求最大值结束--------------*/

	/*----------------OMP加速排序开始----------------*/
	//LARGE_INTEGER  start6 = { 0 }; LARGE_INTEGER  end6 = { 0 };
	//QueryPerformanceCounter(&start6);
	omp_sort(DATANUM, rawFloatData, DATANUM);
	//QueryPerformanceCounter(&end6);
	//QueryPerformanceCounter(&end);
	//std::cout << "OMP加速排序 Time Consumed:" << ((end6.QuadPart - start6.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "OMP加速排序是:";
	//sort_result_check(rawFloatData, DATANUM);
	//std::cout << "从机总共 Time Consumed:" << (end.QuadPart - start.QuadPart) * 1000/ m_liPerfFreq.QuadPart << "ms" << endl;
	//也可以通过以下代码检查正确性，但是会刷屏，拖慢
	/*
	for (int a = 0; a < DATANUM; a++)
		cout << " "<<rawFloatData[a] << " ";
	*/
	/*----------------OMP加速排序结束----------------*/
#endif
	/*-----------------------------------------------------------------*/

#ifdef COMMUNICATION
	/*------------------通信传值前转换--------------------*/
	//sum
#ifdef NOSPEEDUP
	float client_sum[] = { sum_nosp };
#endif
#ifdef FINALSPEEDUP
	float client_sum[] = { sum_sp_avx };
#endif
	unsigned char trans_sum[sizeof(float)];//float有4个字节
	floatArr2charArr(client_sum, 1, trans_sum);

	//max
#ifdef FINALSPEEDUP
	float client_max[] = { max_sp_avx };
#endif
#ifdef NOSPEEDUP
	float client_max[] = { max_nosp };
#endif
	unsigned char trans_max[sizeof(float)];//float有4个字节
	floatArr2charArr(client_max, 1, trans_max);

	//sort
	floatArr2charArr(rawFloatData, DATANUM, trans_sort);
	/*--------------------------------------------------*/
	/*----------------------通信------------------------*/


	//发送
	send(Connection, (char*)trans_sum, 4, NULL);//sum
	
	send(Connection, (char*)trans_max, 4, NULL);//max

	send(Connection, (char*)&trans_sort, 4 * DATANUM, NULL);//sort

	closesocket(Connection);
	WSACleanup();
#endif
	return 0;
}