#define _WINSOCK_DEPRECATED_NO_WARNINGS
/*�³� 1852731 �Զ���*/
/*����� 1951393 �Զ���*/

/*-----socket�õ�-----*/
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
//sse�õ�
#include <immintrin.h>  

using namespace std;

//����ʱ������Ҫ���ĵĺ궨��
#define SPEED 1 //SPEED=1 Ϊ���� SPEED=0Ϊ������
#define PRINT_INFO 0 //1 Ϊ��ӡ��ϸ��Ϣ 0 Ϊ����ӡ��ϸ��Ϣ
#define INET_ADDR "192.168.200.2"//����������Ҫ����
#define HTONS 1111 //�˿ں�
#define MAX_THREADS 64



//˫��ʱ�ܵ�������2000000�����Ե���1000000
//#define SUBDATANUM 2000000
//#define SUBDATANUM 10000
#define SUBDATANUM 1000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*�����ֵ����������*/
#define slice 10000	//ÿ�ν���ֻ����5000�ֽ�(һ���ӽ�������������ᶪ��)
//����������ȡ���ֵ���Կ�һЩ
#define MAX_ONCE(a,b) (((a) > (b)) ? (a):(b))		//max()

//���������ݶ���Ϊ��
float rawFloatData[2 * DATANUM];	//128000000��������Ϊ˫������
float rawFloatData2[2 * DATANUM];	//128000000��������Ϊ˫������
float finalFloatData[2 * DATANUM];	//128000000��������Ϊ˫������
unsigned char recv_sort_char[4*DATANUM];

//float rawFloatData3[DATANUM];
//����Ľ��Ҳ����ȫ�ֱ����У�main�зŲ�����ô���
float sort_nosp[DATANUM];
/*----------------��������----------------*/
//data��ԭʼ���ݣ�lenΪ���ȡ����ͨ����������
float pure_sum(const float data[], const int len); 
//data��ԭʼ���ݣ�lenΪ���ȡ����ͨ����������
float pure_max(const float data[], const int len);
//data��ԭʼ���ݣ�start����ʼλ��,end����ֹλ�á���������result�С�
void pure_sort(const float data[], const int start, const int end, float  result[]);
/*�ж������Ƿ�ɹ�,���ʧ�ܴ�ӡfalse�����ҷ���-2������ɹ���ӡtrue�����ҷ���0*/
int sort_result_check(float* array, int len);
float omp_sum(const float data[], const int len); //data��ԭʼ���ݣ�lenΪ���ȡ����ͨ����������
float omp_max(const float data[], const int len);//data��ԭʼ���ݣ�lenΪ���ȡ����ͨ����������
void omp_sort(const int end, float data[], const int len);//data��ԭʼ���ݣ�lenΪ���ȡ�������ֱ��������RawFloatData��
float sse_sum(float data[], int len);	//data��ԭʼ���ݣ�lenΪ���ȣ�����ֵΪ�ܺ�
float sse_max(float data[], int len);
void floatArr2charArr(float* floatArr, unsigned int len, unsigned char* charArr);/*��������ת�ַ����飬���������ֽ�Ϊ��λ������ת���������lenΪҪת�������ݳ���*/
int recvsuccess = -2;
int received = 0;

/*-----------------------------------------*/

/*----------------�޼����㷨----------------*/
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
//���ù鲢���� ����С����
void pure_sort(float data[], const int start, const int end, float result[])
{
	if (end - start > 1) {
		int m = start + (end - start) / 2;
		int p = start, q = m, i = start;
		pure_sort(data, start, m, result);
		pure_sort(data, m, end, result);

		while (p < m || q < end) {
			if (q >= end || (p < m && log(sqrt(data[p])) <= log(sqrt(data[q])))) {	//�������log(sqrt())����һЩ����ʱ��
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
/*----------------OpenMP����----------------*/
float omp_sum(const float data[], const int len)
{
	double sum = 0.0;
#pragma omp parallel for reduction(+:sum) 
	for (int i = 0; i < len; i++)
		sum += log(sqrt(data[i]));
	return float(sum);
}

float omp_max(const float data[],const int len) //omp�����ֵ
{
	double max_temp = 0;
#pragma omp parallel for
	for (int i = 0; i < len; i++){
//�����ٽ����Ƿ�Ҳ����,�����ٽ������粻��omp
//#pragma omp critical
		max_temp = MAX_ONCE(log(sqrt(data[i]) ), max_temp);
	}
	return float(max_temp);
}

//�ϲ���������
void merge(const int l1, const int r1, const int r2, float data[], float temp[]) {
	int top = l1, p = l1, q = r1;

	while (p < r1 || q < r2) {

		if (q >= r2 || (p < r1 && log(sqrt(data[p])) <= log(sqrt(data[q])))) {	//�������log(sqrt())����һЩ����ʱ��
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
//��������һЩ�Ż���Ԥ����ϲ��˵��������䣬��΢��ߵ��ٶ�
#pragma omp parallel for private(i, t) shared(len, data)
	for (i = 0; i < len / 2; i++)
		if (log(sqrt(data[i * 2])) > log(sqrt(data[i * 2 + 1]))) {
			t = data[i * 2];
			data[i * 2] = data[i * 2 + 1];
			data[i * 2 + 1] = t;
		}
//i����ÿ�ι鲢�����䳤�ȣ�j������Ҫ�鲢��������������С���±�
	for (i = 2; i < end; i *= 2) {
#pragma omp parallel for private(j) shared(end, i)
		for (j = 0; j < end - i; j += i * 2) {
			merge(j, j + i, (j + i * 2 < end ? j + i * 2 : end), data, temp);
		}
	}

	free (temp);
}


/* �鲢���� */
void merge_final(float first[], float second[], float arr[])
{
	int i, j, k;
	const int left=0;
	const int middle= DATANUM-1;
	const int right=2* DATANUM-1;
	
	


	i = 0;
	j = 0;
	k = left;

	/* �Ƚ�������ʱ����������ҳ���ǰ��С������Ȼ������� arr */
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

		k++; /* arr ��������� */
	}

	/* ����ʱ������ʣ��������� arr ���� */
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
/*------------------SSE����------------------*/
float sse_sum(float data[], int len) //��sse�������
{
	__m256* ptr = (__m256*)data; //256bit�������ͣ�8�����и���������  _m256==256λ���������ȣ�AVX��
	__m256 xfsSum = _mm256_setzero_ps();	//Sets float32 YMM registers to zero
	float s = 0;	//����ֵ
	const float* q;	//ָ�봫ֵ��

	for (int i = 0; i < len / 8; ++i, ++ptr)
	{
		__m256 sqr = _mm256_sqrt_ps(*ptr);	//��ȡƽ���������Ӽ���ʱ��
		__m256 lgy = _mm256_log_ps(sqr);	//����log���㣬���Ӽ���ʱ��
		xfsSum = _mm256_add_ps(xfsSum, lgy);//���
	}
	q = (const float*)&xfsSum;
	s = (q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7]);
	return float(s);
}

float sse_max(float data[], int len)
{
	float* result;
	float max_last = 0.0f;
	__m256* ptr = (__m256*)data; //ǿ��ת����m256�������ͣ���8�������������͹���
	__m256 max_temp = _mm256_setzero_ps();
	for (int i = 0; i < len / 8; ++i, ++ptr)
	{
		//__m256 div = _mm256_div_ps(*ptr, _mm256_set1_ps(4.0f));
		__m256 lgy = _mm256_log_ps(_mm256_sqrt_ps(*ptr));
		//__m256 sqrt = _mm256_sqrt_ps(lgy);
		max_temp = _mm256_max_ps(max_temp, lgy);	//ȡ���ֵ
	}
	result = (float*)&max_temp;
	//�ٶ���8��ֵ����������
	for (int i = 0; i < 8; i++)
	{
		max_last = MAX_ONCE(result[i], max_last);
	}
	return max_last;
}
/*------------------------------------------*/

/*�ж������Ƿ�ɹ�,���ʧ�ܴ�ӡfalse�����ҷ���-2������ɹ���ӡtrue�����ҷ���0*/
int sort_result_check(float* array, int len) //�ж������Ƿ�ɹ�
{
	for (int j = 0; j < len - 1; j++)
	{
		if (array[j] > array[j + 1])
		{
			//cout << "j=" << j << endl;
			//cout << "array[j]=" << array[j] << endl;
			//cout << "array[j+1]=" << array[j+1] << endl;
			cout << "�����" << endl;
			return -2;
		}
	}
	cout << "��ȷ��" << endl;
	return 0;
}
/*------------------------Client��------------------------*/
/*��������ת�ַ����飬���������ֽ�Ϊ��λ������ת���������lenΪҪת�������ݳ���*/
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

/*------------------------Server��------------------------*/
//�ַ�ת����(����)
void byteToFloat(unsigned char* charBuf, float* floatout)
{
	unsigned char  i;
	void* pf = floatout;
	unsigned char* px = charBuf;
	for (i = 0; i < 4; i++) {
		*((unsigned char*)pf + i) = *(px + i);
	}
}
//�ַ�����ת�������飬��Ҫ����ת�����յ�������õ�����
void charArr2floatArr(unsigned char* charArrIn, unsigned int len, float* floatArrOut) { //�ַ�����ת��������
	unsigned int floatCnt = 0;
	float* temp = nullptr;
	for (int i = 0; i < len; ) {
		for (int k = 0; k < 4; k++) {
			temp = &floatArrOut[floatCnt];
			*((unsigned char*)temp + k) = *(charArrIn + i);
			i++;
		}
		floatCnt++;//float--4Bytes ÿ���ֽ�++
	}
}
/*--------------------------------------------------------*/
int main()
{
	unsigned char *recv_sort_char2=new unsigned char[4 * DATANUM];
	LARGE_INTEGER m_liPerfFreq = { 0 };
	QueryPerformanceFrequency(&m_liPerfFreq);//��ȡƵ��
	//���ݳ�ʼ��
	for (size_t i = 0; i < DATANUM; i++)//���ݳ�ʼ��
	{
		//rawFloatData[i] = float(i + 1);//������ʼ��ʼ��������
		rawFloatData[i] = float((rand() + rand()));//�����ڴ��ң�����seedһֱ���䣬���Դ�ÿ�δ�slnʱ����Ψһ��
		//rawFloatData2[i] = rawFloatData[i];
		//rawFloatData3[i] = rawFloatData[i];
	}
	float sum_nosp, sum_sp_openmp, sum_sp_sse;
	float max_nosp, max_sp_openmp, max_sp_sse;



/**
* @ ͨ�ų�ʼ��ģ��
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
	addr.sin_port = htons(HTONS); //target Port �������˱���󶨶˿ںţ��ͻ��˲���Ҫ�󶨶˿ںţ��ͻ���Ҫ���߷������Լ��Ķ˿ںţ����ܵõ��������Ļظ�
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
* @ �ȴ�Client����Эͬ����
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
	cout << "Server�յ�CLient���󣬿�ʼЭͬ����" << endl;
	cout << "-----------------------------------" << endl<<endl;
	cout << "-----------����������-----------" << endl << endl;
/** @} */

	LARGE_INTEGER  start = { 0 }; LARGE_INTEGER  end = { 0 };
	QueryPerformanceCounter(&start);

	///*-----------------------------�޼��ٰ汾------------------------------*/

/**
* @ �޼������ģ��
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
	cout << "�޼������ Time Consumed:" << (end1.QuadPart - start1.QuadPart) << endl;
	cout << "�޼�����ͽ��Ϊ:" << sum_nosp << endl;
#endif
#endif
/** @} */
	

/**
* @ �޼��������ֵģ��
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
	cout << "�޼��������ֵ Time Consumed:" << (end2.QuadPart - start2.QuadPart) << endl;
	cout << "�޼��������ֵ���Ϊ:" << max_nosp << endl;
#endif
#endif
/** @} */


/**
* @ �޼�������ģ��
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
	cout << "�޼������� Time Consumed:" << (end3.QuadPart - start3.QuadPart) << endl;
	cout << "�޼���������:";
	sort_result_check(sort_nosp, DATANUM);
#endif
/** @} */


	///*-------------------------------����------------------------------------*/


/**
* @ OMP�������
* @{
*/
/*
	LARGE_INTEGER  start4 = { 0 }; LARGE_INTEGER  end4 = { 0 };
	QueryPerformanceCounter(&start4);
	sum_sp_openmp = omp_sum(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end4);
	cout << "OMP������� Time Consumed:" << (end4.QuadPart - start4.QuadPart) << endl;
	cout << "OMP������ͽ��Ϊ:" << sum_sp_openmp << endl;
*/
/** @} */


/**
* @ OMP���������ֵ
* @{
*/
/*

	LARGE_INTEGER start5 = { 0 }; LARGE_INTEGER end5 = { 0 };
	QueryPerformanceCounter(&start5);
	max_sp_openmp = omp_max(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end5);
	cout << "omp���������ֵ Time Consumed:" << (end5.QuadPart - start5.QuadPart) << endl;
	cout << "omp���������ֵ���Ϊ:" << max_sp_openmp << endl;

* /
/** @} */

/**
* @ SSE�������
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
	cout << "SSE������� Time Consumed:" << (end7.QuadPart - start7.QuadPart) << endl;
	cout << "SSE������ͽ��Ϊ:" << sum_sp_sse << endl;
#endif
#endif
/** @} */

/**
* @ SSE���������ֵ
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
	cout << "SSE���������ֵ Time Consumed:" << (end8.QuadPart - start8.QuadPart) << endl;
	cout << "SSE���������ֵ���Ϊ:" << max_sp_sse << endl;
#endif
#endif
/** @} */
	

/**
* @ OMP��������
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
	cout << "OMP�������� Time Consumed:" << (end6.QuadPart - start6.QuadPart) << endl;
	cout << "OMP����������:";
	sort_result_check(rawFloatData, DATANUM);
#endif
/** @} */



/**
* @˫������������
*
* ����ͨ�ţ����մ�client�������ļ�����
* ��˫�������һ������ݽ�����������
* �����������: ��˫���ֱ���͵Ľ�����
* ���ֵ��������: ��˫���ֱ��������ֵ��ȡ���
* ������������: ��˫������鲢
* 
* @{
*/
	unsigned char recv_sum_char[4];//���յ���raw���
	unsigned char recv_max_char[4];//���յ���raw���ֵ
	float recv_sum_float = 0.0f;//ת����ĵ�float���
	float recv_max_float = 0.0f;//ת����ĵ�float���ֵ
	unsigned char* ptr = recv_sort_char;

	//���պ�
	recv(newConnection, (char*)&recv_sum_char, 4, NULL);
	byteToFloat(recv_sum_char, &recv_sum_float);
	//�������ֵ
	recv(newConnection, (char*)&recv_max_char, 4, NULL);
	byteToFloat(recv_max_char, &recv_max_float);
	//��������������
	bytes = 0;
	//�ֿ������գ�����̫��
	while (bytes < 4* DATANUM){
		recvsuccess = recv(newConnection, (char*)&ptr[bytes], slice, NULL);
		bytes += recvsuccess;
	}


	//���յ������������õ�rawFloatData�ĵ�DataNum��
	charArr2floatArr(recv_sort_char, 4 * DATANUM, &rawFloatData[DATANUM]);
	QueryPerformanceCounter(&end_tx);
#if PRINT_INFO
	printf("Client�ѷ���������Server");
#endif


#if SPEED 
	//˫�����
	float total_sum = sum_sp_sse + recv_sum_float;
	//˫���������ֵ
	float final_max = MAX_ONCE(max_sp_sse, recv_max_float);
#else
	//˫�����
	float total_sum = sum_nosp + recv_sum_float;
	//˫���������ֵ
	float final_max = MAX_ONCE(max_nosp, recv_max_float);
#endif
	//˫������鲢
	merge_final(rawFloatData, &(rawFloatData[DATANUM]), finalFloatData);
/** @} */


	QueryPerformanceCounter(&end);

	cout << "������ͽ��: " << total_sum << endl;
	cout << "�������ֵΪ: " << final_max << endl;
	cout << "������������:";
	sort_result_check(finalFloatData, 2 * DATANUM);
	cout << "ͨ��ʱ�� Time Consumed:" << (end_tx.QuadPart - start_tx.QuadPart) * 1000 / m_liPerfFreq.QuadPart << "ms" << endl;
	cout << "˫���ܹ� Time Consumed:" << (end.QuadPart - start.QuadPart) * 1000 / m_liPerfFreq.QuadPart <<"ms" << endl;
	cout << "����Эͬ�������" << endl << endl;

	
	}
	return 0;
}