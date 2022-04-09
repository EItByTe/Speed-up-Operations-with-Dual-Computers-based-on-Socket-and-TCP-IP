#define _WINSOCK_DEPRECATED_NO_WARNINGS
/*�³� 1852731 �Զ���*/
/*����� 1951393 �Զ���*/
/*-----socket�õ�-----*/
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

#define INET_ADDR "192.168.200.2"//����������Ҫ���ģ�zrg���ֻ��ȵ�
#define MAX_THREADS 64
//˫��ʱ�ܵ�������2000000�����Ե���1000000
//#define SUBDATANUM 2000000
#define SUBDATANUM 1000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*�����ֵ����������*/

//����������ȡ���ֵ���Կ�һЩ
#define MAX_ONCE(a,b) (((a) > (b)) ? (a):(b))		//max()

//���������ݶ���Ϊ��
float rawFloatData[DATANUM];
float rawIntData[DATANUM];
unsigned char trans_sort[4*DATANUM];
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
float avx_sum(float data[], int len);	//data��ԭʼ���ݣ�lenΪ���ȣ�����ֵΪ�ܺ�
float avx_max(float data[], int len);
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
		if (log(sqrt(data[i])) > max_temp)
			max_temp = log(sqrt(data[i]));
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
	double sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < len; i++)
		sum += log(sqrt(data[i]));
	return float(sum);
}

float omp_max(const float data[],const int len) //omp�����ֵ
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
}
/*------------------------------------------*/
/*------------------AVX����------------------*/
float avx_sum(float data[], int len) //��AVX�������
{
	__m256* ptr = (__m256*)data; //256bit�������ͣ�8�����и���������  _m256==256λ���������ȣ�AVX��
	__m256 xfsSum = _mm256_setzero_ps();	//Sets float32 YMM registers to zero
	double s = 0;	//����ֵ
	const float* q;	//ָ�봫ֵ��
	for (int i = 0; i < len / 8; ++i, ++ptr)
	{
		//__m256 sqr = _mm256_sqrt_ps(*ptr);	//��ȡƽ���������Ӽ���ʱ��
		//__m256 lgy = _mm256_log_ps(sqr);	//����log���㣬���Ӽ���ʱ��
		xfsSum = _mm256_add_ps(xfsSum, _mm256_log_ps(_mm256_sqrt_ps(*ptr)));//���
	}
	q = (const float*)&xfsSum;
	s = (q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7]);
	return float(s);
}

float avx_max(float data[], int len)
{
	float* result;
	float max_last = 0.0f;
	__m256* ptr = (__m256*)data; //ǿ��ת����m256�������ͣ���8�������������͹���
	__m256 max_temp = _mm256_setzero_ps();
//#pragma omp parallel for
	for (int i = 0; i < len / 8; ++i)
	{
		__m256 lgy = _mm256_log_ps(_mm256_sqrt_ps(*ptr));
		max_temp = _mm256_max_ps(max_temp, lgy);	//ȡ���ֵ
		++ptr;
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
/*------------------AVX��OMP����------------------*/
float avx_omp_sum(float data[], int len) //��avx�������
{
	__m256* ptr = (__m256*)data; //256bit�������ͣ�8�����и���������  _m256==256λ���������ȣ�AVX��
	//__m256 xfsSum = _mm256_setzero_ps();	//Sets float32 YMM registers to zero
	double sum = 0;	//����ֵ
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < len / 8; i++)
	{
		//__m256 sqr = _mm256_sqrt_ps(*ptr);	//��ȡƽ���������Ӽ���ʱ��
		__m256 xfsSum = _mm256_log_ps(_mm256_sqrt_ps(_mm256_loadu_ps(data)));	//����log���㣬���Ӽ���ʱ��
		///xfsSum = _mm256_add_ps(xfsSum, lgy);//���
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
	//�ٶ���8��ֵ����������
	for (int i = 0; i < MAX_THREADS; i++){
		max_last = MAX_ONCE(max_temp_avx[i], max_last);
	}
	return float(max_last);
}
/*�ж������Ƿ�ɹ�,���ʧ�ܴ�ӡfalse�����ҷ���-2������ɹ���ӡtrue�����ҷ���0*/
int sort_result_check(float* array, int len) //�ж������Ƿ�ɹ�
{
	for (int j = 0; j < len - 1; j++)
	{
		if (array[j] > array[j + 1])
		{
			std::cout << "�����" << endl;
			return -2;
		}
	}
	std::cout << "��ȷ��" << endl;
	return 0;
}

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
/*������*/
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
	/*-------------------------ͨ�ų�ʼ��------------------------*/
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
	addr.sin_port = htons(1111); //target Port �������˱���󶨶˿ںţ��ͻ��˲���Ҫ�󶨶˿ںţ��ͻ���Ҫ���߷������Լ��Ķ˿ںţ����ܵõ��������Ļظ�
	addr.sin_family = AF_INET; //IPv4 Socket

	SOCKET Connection = socket(AF_INET, SOCK_STREAM, NULL); //stream�������ٺ�����
	if (connect(Connection, (SOCKADDR*)&addr, sizeofaddr) != 0) //Connection
	{
		MessageBoxA(NULL, "Bad Connection", "Error", MB_OK | MB_ICONERROR);
		return 0;
	}
	srand(time(NULL));
#endif
	/*----------------------------------------------------------*/
	LARGE_INTEGER m_liPerfFreq = { 0 };
	QueryPerformanceFrequency(&m_liPerfFreq);//��ȡƵ��
	//���ݳ�ʼ��
	for (size_t i = 0; i < DATANUM; i++)//���ݳ�ʼ��
	{
		//rawFloatData[i] = float(i + 1);//������ʼ��ʼ��������
		rawFloatData[i] = float((rand() + rand()));//�����ڴ��ң�����seedһֱ���䣬���Դ�ÿ�δ�slnʱ����Ψһ��
	}
	float sum_nosp, sum_sp_openmp, sum_sp_avx;
	float max_nosp, max_sp_openmp, max_sp_avx;

#ifdef COMMUNICATION
	char start_flag = 2;
	/*�ӻ��˳�ʼ����ϣ����ͱ�־λ���÷������˿�ʼ���㡢��ʱ*/
	cout << "�ӻ���������Эͬ���� " << endl;
	int bytes_length = send(Connection, (char*)&start_flag, 1, NULL);//flag
#endif
	//LARGE_INTEGER  start = { 0 }; LARGE_INTEGER  end = { 0 };
	//QueryPerformanceCounter(&start);
	///*-----------------------------�޼��ٰ汾------------------------------*/
#ifdef NOSPEEDUP
	/*----------------�޼�����Ϳ�ʼ---------------*/
	LARGE_INTEGER  start1 = { 0 }; LARGE_INTEGER  end1 = { 0 };
	QueryPerformanceCounter(&start1);
	sum_nosp = pure_sum(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end1);
	std::cout << "�޼������ Time Consumed:" << ((end1.QuadPart - start1.QuadPart) * 1000/ m_liPerfFreq.QuadPart) <<"ms"<< endl;
	std::cout << "�޼�����ͽ��Ϊ:" << sum_nosp << endl;
	/*----------------�޼�����ͽ���---------------*/

	/*--------------�޼��������ֵ��ʼ--------------*/
	LARGE_INTEGER  start2 = { 0 }; LARGE_INTEGER  end2 = { 0 };
	QueryPerformanceCounter(&start2);
	max_nosp = pure_max(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end2);
	std::cout << "�޼��������ֵ Time Consumed:" << ((end2.QuadPart - start2.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	std::cout << "�޼��������ֵ���Ϊ:" << max_nosp << endl;
	/*--------------�޼��������ֵ����--------------*/

	/*----------------�޼�������ʼ----------------*/
	LARGE_INTEGER  start3 = { 0 }; LARGE_INTEGER  end3 = { 0 };
	QueryPerformanceCounter(&start3);
	pure_sort(rawFloatData, 0, DATANUM, sort_nosp);
	QueryPerformanceCounter(&end3);
	////Ҳ����ͨ�����´�������ȷ�ԣ����ǻ�ˢ��������
	/*
	for (int a = 0; a < DATANUM; a++)
		cout << rawFloatData[a] << " ";
	*/
	std::cout << "�޼������� Time Consumed:" << ((end3.QuadPart - start3.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	std::cout << "�޼���������:";
	sort_result_check(sort_nosp, DATANUM);
	///*----------------�޼����������----------------*/
#endif
	///*---------------------------------------------------------------------*/
	/*-----------------------------OpenMP����------------------------------*/

	/*----------------OMP������Ϳ�ʼ---------------*/
	//LARGE_INTEGER  start4 = { 0 }; LARGE_INTEGER  end4 = { 0 };
	//QueryPerformanceCounter(&start4);
	//sum_sp_openmp = omp_sum(rawFloatData, DATANUM);
	//QueryPerformanceCounter(&end4);
	//std::cout << "OMP������� Time Consumed:" << ((end4.QuadPart - start4.QuadPart)  * 1000/ m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "OMP������ͽ��Ϊ:" << sum_sp_openmp << endl;
	/*----------------OMP������ͽ���---------------*/

	///*--------------OMP���������ֵ��ʼ--------------*/
	//LARGE_INTEGER start5 = { 0 }; LARGE_INTEGER end5 = { 0 };
	//QueryPerformanceCounter(&start5);
	//max_sp_openmp = omp_max(rawFloatData, DATANUM);
	//QueryPerformanceCounter(&end5);
	//std::cout << "omp���������ֵ Time Consumed:" << ((end5.QuadPart - start5.QuadPart) * 1000/ m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "omp���������ֵ���Ϊ:" << max_sp_openmp << endl;
	///*--------------OMP���������ֵ����--------------*/

	/*---------------------------------------------------------------------*/

	/*-----------------------------AVX����------------------------------*/
#ifdef FINALSPEEDUP
	/*----------------AVX������Ϳ�ʼ----------------*/
	//LARGE_INTEGER  start7 = { 0 }; LARGE_INTEGER  end7 = { 0 };
	//QueryPerformanceCounter(&start7);
	sum_sp_avx = avx_sum(rawFloatData, DATANUM);
	//sum_sp_avx = avx_omp_sum(rawFloatData, DATANUM);	//avx+omp
	//QueryPerformanceCounter(&end7);    
	//std::cout << "AVX������� Time Consumed:" << ((end7.QuadPart - start7.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "AVX������ͽ��Ϊ:" << sum_sp_avx << endl;
	/*----------------AVX������ͽ���----------------*/

	/*--------------AVX���������ֵ��ʼ--------------*/
	//LARGE_INTEGER start8 = { 0 }; LARGE_INTEGER end8 = { 0 };
	//QueryPerformanceCounter(&start8);
	max_sp_avx = avx_max(rawFloatData, DATANUM);
	//max_sp_avx = avx_omp_max(rawFloatData, DATANUM); //avx+omp
	//QueryPerformanceCounter(&end8);
	//std::cout << "AVX���������ֵ Time Consumed:" << ((end8.QuadPart - start8.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "AVX���������ֵ���Ϊ:" << max_sp_avx << endl;
	/*--------------AVX���������ֵ����--------------*/

	/*----------------OMP��������ʼ----------------*/
	//LARGE_INTEGER  start6 = { 0 }; LARGE_INTEGER  end6 = { 0 };
	//QueryPerformanceCounter(&start6);
	omp_sort(DATANUM, rawFloatData, DATANUM);
	//QueryPerformanceCounter(&end6);
	//QueryPerformanceCounter(&end);
	//std::cout << "OMP�������� Time Consumed:" << ((end6.QuadPart - start6.QuadPart) * 1000 / m_liPerfFreq.QuadPart) << "ms" << endl;
	//std::cout << "OMP����������:";
	//sort_result_check(rawFloatData, DATANUM);
	//std::cout << "�ӻ��ܹ� Time Consumed:" << (end.QuadPart - start.QuadPart) * 1000/ m_liPerfFreq.QuadPart << "ms" << endl;
	//Ҳ����ͨ�����´�������ȷ�ԣ����ǻ�ˢ��������
	/*
	for (int a = 0; a < DATANUM; a++)
		cout << " "<<rawFloatData[a] << " ";
	*/
	/*----------------OMP�����������----------------*/
#endif
	/*-----------------------------------------------------------------*/

#ifdef COMMUNICATION
	/*------------------ͨ�Ŵ�ֵǰת��--------------------*/
	//sum
#ifdef NOSPEEDUP
	float client_sum[] = { sum_nosp };
#endif
#ifdef FINALSPEEDUP
	float client_sum[] = { sum_sp_avx };
#endif
	unsigned char trans_sum[sizeof(float)];//float��4���ֽ�
	floatArr2charArr(client_sum, 1, trans_sum);

	//max
#ifdef FINALSPEEDUP
	float client_max[] = { max_sp_avx };
#endif
#ifdef NOSPEEDUP
	float client_max[] = { max_nosp };
#endif
	unsigned char trans_max[sizeof(float)];//float��4���ֽ�
	floatArr2charArr(client_max, 1, trans_max);

	//sort
	floatArr2charArr(rawFloatData, DATANUM, trans_sort);
	/*--------------------------------------------------*/
	/*----------------------ͨ��------------------------*/


	//����
	send(Connection, (char*)trans_sum, 4, NULL);//sum
	
	send(Connection, (char*)trans_max, 4, NULL);//max

	send(Connection, (char*)&trans_sort, 4 * DATANUM, NULL);//sort

	closesocket(Connection);
	WSACleanup();
#endif
	return 0;
}