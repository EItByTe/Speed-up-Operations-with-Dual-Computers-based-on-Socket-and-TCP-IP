#define _WINSOCK_DEPRECATED_NO_WARNINGS
/*陈晨 1852731 自动化*/
/*张儒戈 1951393 自动化*/

/*-----socket用到-----*/
#pragma comment(lib,"ws2_32.lib")
#include <WinSock2.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
/*------------------*/
#include <iostream>
#include <math.h>
#include <omp.h>		//add for omp
#include <Windows.h>	//add for taking frequence
//sse用到
#include <immintrin.h>  

using namespace std;

//复现时可能需要更改的宏定义
#define SPEED 1 //SPEED=1 为加速 SPEED=0为不加速
#define PRINT_INFO 0 //1 为打印详细信息 0 为不打印详细信息
#define INET_ADDR "192.168.200.2"//更换网络需要更改
#define HTONS 1111 //端口号
#define MAX_THREADS 64



//双机时总的数据量2000000，所以单机1000000
//#define SUBDATANUM 2000000
//#define SUBDATANUM 10000
#define SUBDATANUM 1000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*这个数值是总数据量*/
#define slice 10000	//每次接收只接收5000字节(一下子接收数据量过大会丢包)
//最快的两个数取最大值，稍快一些
#define MAX_ONCE(a,b) (((a) > (b)) ? (a):(b))		//max()

//待测试数据定义为：
float rawFloatData[2 * DATANUM];	//128000000数据总量为双机总量
float rawFloatData2[2 * DATANUM];	//128000000数据总量为双机总量
float finalFloatData[2 * DATANUM];	//128000000数据总量为双机总量
unsigned char recv_sort_char[4*DATANUM];

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
float sse_sum(float data[], int len);	//data是原始数据，len为长度，返回值为总和
float sse_max(float data[], int len);
void floatArr2charArr(float* floatArr, unsigned int len, unsigned char* charArr);/*浮点数组转字符数组，发送是以字节为单位，若不转换，会出错，len为要转换的数据长度*/
int recvsuccess = -2;
int received = 0;

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
		max_temp = MAX_ONCE(log(sqrt(data[i])), max_temp);
		//if (log(sqrt(data[i]))> max)
		//	max = log(sqrt(data[i]));
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
	double sum = 0.0;
#pragma omp parallel for reduction(+:sum) 
	for (int i = 0; i < len; i++)
		sum += log(sqrt(data[i]));
	return float(sum);
}

float omp_max(const float data[],const int len) //omp求最大值
{
	double max_temp = 0;
#pragma omp parallel for
	for (int i = 0; i < len; i++){
//不加临界区是否也可以,加了临界区不如不用omp
//#pragma omp critical
		max_temp = MAX_ONCE(log(sqrt(data[i]) ), max_temp);
	}
	return float(max_temp);
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

	free (temp);
}


/* 归并函数 */
void merge_final(float first[], float second[], float arr[])
{
	int i, j, k;
	const int left=0;
	const int middle= DATANUM-1;
	const int right=2* DATANUM-1;
	
	


	i = 0;
	j = 0;
	k = left;

	/* 比较两个临时数组的数，找出当前最小的数，然后按序存入 arr */
	while (i < DATANUM && j < DATANUM)
	{

		if (first[i] <= second[j])
		{
			arr[k] = first[i];
			++i;
		}
		else
		{
			arr[k] = second[j];
			j++;
		}

		k++; /* arr 数组的索引 */
	}

	/* 将临时数组中剩余的数存入 arr 数组 */
	while (i < DATANUM)
	{
		arr[k] = first[i];
		i++;
		k++;
	}

	while (j < DATANUM)
	{
		arr[k] = second[j];
		j++;
		k++;
	}
}


/*------------------------------------------*/
/*------------------SSE加速------------------*/
float sse_sum(float data[], int len) //用sse加速求和
{
	__m256* ptr = (__m256*)data; //256bit数据类型，8个并行浮点数构成  _m256==256位紧缩单精度（AVX）
	__m256 xfsSum = _mm256_setzero_ps();	//Sets float32 YMM registers to zero
	float s = 0;	//返回值
	const float* q;	//指针传值用

	for (int i = 0; i < len / 8; ++i, ++ptr)
	{
		__m256 sqr = _mm256_sqrt_ps(*ptr);	//先取平方根，增加计算时间
		__m256 lgy = _mm256_log_ps(sqr);	//再做log运算，增加计算时间
		xfsSum = _mm256_add_ps(xfsSum, lgy);//求和
	}
	q = (const float*)&xfsSum;
	s = (q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7]);
	return float(s);
}

float sse_max(float data[], int len)
{
	float* result;
	float max_last = 0.0f;
	__m256* ptr = (__m256*)data; //强制转换成m256数据类型，由8个并行数据类型构成
	__m256 max_temp = _mm256_setzero_ps();
	for (int i = 0; i < len / 8; ++i, ++ptr)
	{
		//__m256 div = _mm256_div_ps(*ptr, _mm256_set1_ps(4.0f));
		__m256 lgy = _mm256_log_ps(_mm256_sqrt_ps(*ptr));
		//__m256 sqrt = _mm256_sqrt_ps(lgy);
		max_temp = _mm256_max_ps(max_temp, lgy);	//取最大值
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

/*判断排序是否成功,如果失败打印false，并且返回-2，如果成功打印true，并且返回0*/
int sort_result_check(float* array, int len) //判断排序是否成功
{
	for (int j = 0; j < len - 1; j++)
	{
		if (array[j] > array[j + 1])
		{
			//cout << "j=" << j << endl;
			//cout << "array[j]=" << array[j] << endl;
			//cout << "array[j+1]=" << array[j+1] << endl;
			cout << "错误的" << endl;
			return -2;
		}
	}
	cout << "正确的" << endl;
	return 0;
}
/*------------------------Client用------------------------*/
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
/*--------------------------------------------------------*/

/*------------------------Server用------------------------*/
//字符转浮点(单个)
void byteToFloat(unsigned char* charBuf, float* floatout)
{
	unsigned char  i;
	void* pf = floatout;
	unsigned char* px = charBuf;
	for (i = 0; i < 4; i++) {
		*((unsigned char*)pf + i) = *(px + i);
	}
}
//字符数组转浮点数组，主要用于转换接收到的排序好的数组
void charArr2floatArr(unsigned char* charArrIn, unsigned int len, float* floatArrOut) { //字符数组转浮点数组
	unsigned int floatCnt = 0;
	float* temp = nullptr;
	for (int i = 0; i < len; ) {
		for (int k = 0; k < 4; k++) {
			temp = &floatArrOut[floatCnt];
			*((unsigned char*)temp + k) = *(charArrIn + i);
			i++;
		}
		floatCnt++;//float--4Bytes 每四字节++
	}
}
/*--------------------------------------------------------*/
int main()
{
	unsigned char *recv_sort_char2=new unsigned char[4 * DATANUM];
	LARGE_INTEGER m_liPerfFreq = { 0 };
	QueryPerformanceFrequency(&m_liPerfFreq);//获取频率
	//数据初始化
	for (size_t i = 0; i < DATANUM; i++)//数据初始化
	{
		//rawFloatData[i] = float(i + 1);//这样初始就始终有序了
		rawFloatData[i] = float((rand() + rand()));//类似于打乱，但是seed一直不变，所以答案每次打开sln时都是唯一的
		//rawFloatData2[i] = rawFloatData[i];
		//rawFloatData3[i] = rawFloatData[i];
	}
	float sum_nosp, sum_sp_openmp, sum_sp_sse;
	float max_nosp, max_sp_openmp, max_sp_sse;



/**
* @ 通信初始化模块
* @{
*/
	WSAData wsaData;
	WORD DllVersion = MAKEWORD(2, 1);
	if (WSAStartup(DllVersion, &wsaData) != 0)
	{
		MessageBoxA(NULL, "WinSock startup error", "Error", MB_OK | MB_ICONERROR);
		return 0;
	}
	SOCKADDR_IN addr;
	int addrlen = sizeof(addr);
	addr.sin_addr.s_addr = inet_addr(INET_ADDR); //target PC
	addr.sin_port = htons(HTONS); //target Port 服务器端必须绑定端口号，客户端不需要绑定端口号，客户端要告诉服务器自己的端口号，才能得到服务器的回复
	addr.sin_family = AF_INET; //IPv4 Socket

	SOCKET sListen = socket(AF_INET, SOCK_STREAM, NULL);

	fd_set fdSocket;
	bind(sListen, (SOCKADDR*)&addr, sizeof(addr));
	listen(sListen, SOMAXCONN);
	sockaddr_in addrRemote;
	int nAddrLen = sizeof(addrRemote);

/** @} */ 

	while(1){

/**
* @ 等待Client请求协同计算
* @{
*/
	printf("Waiting for CLient___\n");
	LARGE_INTEGER  start_tx = { 0 }; LARGE_INTEGER  end_tx = { 0 };
	QueryPerformanceCounter(&start_tx);


	SOCKET newConnection = accept(sListen, (sockaddr*)&addrRemote, &nAddrLen);

	int ClientRequire = 0;
	int bytes = 0;


	while (ClientRequire != 2 ) {
		recv(newConnection, (char*)&ClientRequire, 1, NULL);
		//ClientRequire += recvsuccess;
		
	}
	printf("\n\n");
	cout << "-----------------------------------" << endl;
	cout << "Server收到CLient请求，开始协同计算" << endl;
	cout << "-----------------------------------" << endl<<endl;
	cout << "-----------计算结果如下-----------" << endl << endl;
/** @} */

	LARGE_INTEGER  start = { 0 }; LARGE_INTEGER  end = { 0 };
	QueryPerformanceCounter(&start);

	///*-----------------------------无加速版本------------------------------*/

/**
* @ 无加速求和模块
* @{
*/


#if !SPEED
#if PRINT_INFO
	LARGE_INTEGER  start1 = { 0 }; LARGE_INTEGER  end1 = { 0 };
	QueryPerformanceCounter(&start1);
#endif


	sum_nosp = pure_sum(rawFloatData, DATANUM);


#if PRINT_INFO
	QueryPerformanceCounter(&end1);
	cout << "无加速求和 Time Consumed:" << (end1.QuadPart - start1.QuadPart) << endl;
	cout << "无加速求和结果为:" << sum_nosp << endl;
#endif
#endif
/** @} */
	

/**
* @ 无加速求最大值模块
* @{
*/
#if !SPEED
#if PRINT_INFO
	LARGE_INTEGER  start2 = { 0 }; LARGE_INTEGER  end2 = { 0 };
	QueryPerformanceCounter(&start2);
#endif


	max_nosp = pure_max(rawFloatData, DATANUM);


#if PRINT_INFO
	QueryPerformanceCounter(&end2);
	cout << "无加速求最大值 Time Consumed:" << (end2.QuadPart - start2.QuadPart) << endl;
	cout << "无加速求最大值结果为:" << max_nosp << endl;
#endif
#endif
/** @} */


/**
* @ 无加速排序模块
* @{
*/
#if PRINT_INFO
	LARGE_INTEGER  start3 = { 0 }; LARGE_INTEGER  end3 = { 0 };
	QueryPerformanceCounter(&start3);
#endif

#if !SPEED
	pure_sort(rawFloatData, 0, DATANUM, sort_nosp);
#endif

#if PRINT_INFO
	QueryPerformanceCounter(&end3);
	cout << "无加速排序 Time Consumed:" << (end3.QuadPart - start3.QuadPart) << endl;
	cout << "无加速排序是:";
	sort_result_check(sort_nosp, DATANUM);
#endif
/** @} */


	///*-------------------------------加速------------------------------------*/


/**
* @ OMP加速求和
* @{
*/
/*
	LARGE_INTEGER  start4 = { 0 }; LARGE_INTEGER  end4 = { 0 };
	QueryPerformanceCounter(&start4);
	sum_sp_openmp = omp_sum(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end4);
	cout << "OMP加速求和 Time Consumed:" << (end4.QuadPart - start4.QuadPart) << endl;
	cout << "OMP加速求和结果为:" << sum_sp_openmp << endl;
*/
/** @} */


/**
* @ OMP加速求最大值
* @{
*/
/*

	LARGE_INTEGER start5 = { 0 }; LARGE_INTEGER end5 = { 0 };
	QueryPerformanceCounter(&start5);
	max_sp_openmp = omp_max(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end5);
	cout << "omp加速求最大值 Time Consumed:" << (end5.QuadPart - start5.QuadPart) << endl;
	cout << "omp加速求最大值结果为:" << max_sp_openmp << endl;

* /
/** @} */

/**
* @ SSE加速求和
* @{
*/
#if SPEED
#if PRINT_INFO
	LARGE_INTEGER  start7 = { 0 }; LARGE_INTEGER  end7 = { 0 };
	QueryPerformanceCounter(&start7);
#endif


	sum_sp_sse = sse_sum(rawFloatData, DATANUM);


#if PRINT_INFO
	QueryPerformanceCounter(&end7);
	cout << "SSE加速求和 Time Consumed:" << (end7.QuadPart - start7.QuadPart) << endl;
	cout << "SSE加速求和结果为:" << sum_sp_sse << endl;
#endif
#endif
/** @} */

/**
* @ SSE加速求最大值
* @{
*/	
#if SPEED
#if PRINT_INFO
	LARGE_INTEGER start8 = { 0 }; LARGE_INTEGER end8 = { 0 };
	QueryPerformanceCounter(&start8);
#endif

	max_sp_sse = sse_max(rawFloatData, DATANUM);

#if PRINT_INFO
	QueryPerformanceCounter(&end8);
	cout << "SSE加速求最大值 Time Consumed:" << (end8.QuadPart - start8.QuadPart) << endl;
	cout << "SSE加速求最大值结果为:" << max_sp_sse << endl;
#endif
#endif
/** @} */
	

/**
* @ OMP加速排序
* @{
*/
#if PRINT_INFO
	LARGE_INTEGER  start6 = { 0 }; LARGE_INTEGER  end6 = { 0 };
	QueryPerformanceCounter(&start6);
#endif


#if SPEED 
	omp_sort(DATANUM, rawFloatData, DATANUM);
#endif

#if PRINT_INFO
	QueryPerformanceCounter(&end6);
	cout << "OMP加速排序 Time Consumed:" << (end6.QuadPart - start6.QuadPart) << endl;
	cout << "OMP加速排序是:";
	sort_result_check(rawFloatData, DATANUM);
#endif
/** @} */



/**
* @双机计算结果整合
*
* 进行通信，接收从client发送来的计算结果
* 将双机计算的一半的数据进行最终整合
* 求和最终整合: 将双机分别求和的结果相加
* 最大值最终整合: 对双机分别计算的最大值求取最大
* 排序最终整合: 对双机排序归并
* 
* @{
*/
	unsigned char recv_sum_char[4];//接收到的raw求和
	unsigned char recv_max_char[4];//接收到的raw最大值
	float recv_sum_float = 0.0f;//转换后的的float求和
	float recv_max_float = 0.0f;//转换后的的float最大值
	unsigned char* ptr = recv_sort_char;

	//接收和
	recv(newConnection, (char*)&recv_sum_char, 4, NULL);
	byteToFloat(recv_sum_char, &recv_sum_float);
	//接收最大值
	recv(newConnection, (char*)&recv_max_char, 4, NULL);
	byteToFloat(recv_max_char, &recv_max_float);
	//接收排序后的数组
	bytes = 0;
	//分开来接收，否则太多
	while (bytes < 4* DATANUM){
		recvsuccess = recv(newConnection, (char*)&ptr[bytes], slice, NULL);
		bytes += recvsuccess;
	}


	//把收到的排序结果放置到rawFloatData的第DataNum后
	charArr2floatArr(recv_sort_char, 4 * DATANUM, &rawFloatData[DATANUM]);
	QueryPerformanceCounter(&end_tx);
#if PRINT_INFO
	printf("Client已发送数据至Server");
#endif


#if SPEED 
	//双机求和
	float total_sum = sum_sp_sse + recv_sum_float;
	//双机最终最大值
	float final_max = MAX_ONCE(max_sp_sse, recv_max_float);
#else
	//双机求和
	float total_sum = sum_nosp + recv_sum_float;
	//双机最终最大值
	float final_max = MAX_ONCE(max_nosp, recv_max_float);
#endif
	//双机排序归并
	merge_final(rawFloatData, &(rawFloatData[DATANUM]), finalFloatData);
/** @} */


	QueryPerformanceCounter(&end);

	cout << "最终求和结果: " << total_sum << endl;
	cout << "最终最大值为: " << final_max << endl;
	cout << "最终排序结果是:";
	sort_result_check(finalFloatData, 2 * DATANUM);
	cout << "通信时间 Time Consumed:" << (end_tx.QuadPart - start_tx.QuadPart) * 1000 / m_liPerfFreq.QuadPart << "ms" << endl;
	cout << "双机总共 Time Consumed:" << (end.QuadPart - start.QuadPart) * 1000 / m_liPerfFreq.QuadPart <<"ms" << endl;
	cout << "本次协同计算结束" << endl << endl;

	
	}
	return 0;
}