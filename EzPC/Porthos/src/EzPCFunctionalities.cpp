/*

Authors: Nishant Kumar, Mayank Rathee.

Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#pragma once
#include "EzPCFunctionalities.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <thread>
#include <numeric>



static std::vector<double> segmentCpuUsages;
static int numCores = std::thread::hardware_concurrency();
auto globalStartWallTime = std::chrono::high_resolution_clock::now();
static double peakCpuUsage = 0;


extern int partyNum;
extern CommunicationObject commObject;

using namespace std::chrono;

// Function to get CPU time used by the current process
long GetCpuTime() {
    std::ifstream procStat("/proc/self/stat");
    std::string line;
    std::getline(procStat, line);
    std::istringstream iss(line);

    std::vector<std::string> stats;
    std::string stat;
    while (iss >> stat) {
        stats.push_back(stat);
    }

    // The 14th and 15th values are utime and stime respectively
    long utime = std::stol(stats[13]);
    long stime = std::stol(stats[14]);

    return utime + stime;
}
//std::chrono::high_resolution_clock::time_point globalStartWallTime = std::chrono::high_resolution_clock::now();
long globalStartCpuTime = GetCpuTime();  // This should be set in a function if it depends on runtime initialization
void CaptureCpuUsage() {
    auto currentWallTime = std::chrono::high_resolution_clock::now();
    long currentCpuTime = GetCpuTime();

//    Calculate elapsed times for this segment
    std::chrono::duration<double> elapsedWallTime = currentWallTime - globalStartWallTime;
    double elapsedCpuTime = static_cast<double>(currentCpuTime - globalStartCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);

    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);
}


/***************** Functions for different layers in NN *********************/
void MatMul2D(int32_t i, 
		int32_t j, 
		int32_t k, 
		vector< vector<porthosSecretType> >& A, 
		vector< vector<porthosSecretType> >& B, 
		vector< vector<porthosSecretType> >& C, 
		bool modelIsA)
{
    
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	log_print("EzPCFunctionalities : Starting MatMulCSF2D ... ");

//#if (LOG_LAYERWISE)	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
//#endif

	vector<porthosSecretType> X(i*j);
	vector<porthosSecretType> Y(j*k);
	vector<porthosSecretType> Z(i*k);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<j; jj++){
			X[ii*j + jj] = A[ii][jj]; //Each row is of size j
		}
	}
	for (int ii=0; ii<j; ii++){
		for (int jj=0; jj<k; jj++){
			Y[ii*k + jj] = B[ii][jj]; //Each row is of size k
		}
	}
	funcMatMulMPC(X, Y, Z, i, j, k, 0, 0, 0, false);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<k; jj++){
			C[ii][jj] = Z[ii*k + jj]; //Each row is of size k
		}
	}

//#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeMatmul[1] += tt;
	commObject.dataMatmul[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataMatmul[1] += (commObject.totalDataReceived - bytesReceived);
//#endif
    
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

        // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

        // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
    cout<<"cpu uage matmul: "<<cpuUsage<<endl;
    CaptureCpuUsage();
}

void ArgMax(int32_t s1, 
		int32_t s2, 
		vector< vector<porthosSecretType> >& inArr, 
		vector<porthosSecretType>& outArr)
{
    
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();

		
	log_print("EzPCFunctionalities : Starting ArgMax1 ... ");
       high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
	vector<porthosSecretType> Arr(s1*s2);
	vector<porthosSecretType> maxi(s1);
	for(int ii=0;ii<s1;ii++){
		for(int jj=0;jj<s2;jj++){
			Arr[ii*s2 + jj] = inArr[ii][jj]; //Each row is of size inArrS2
		}
	}
	funcMaxMPC(Arr, maxi, outArr, s1, s2, true);
       	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeArgmax += tt;
	commObject.dataArgmax[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataArgmax[1] += (commObject.totalDataReceived - bytesReceived);
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage argmax: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}

void Relu(int32_t size, 
		vector<porthosSecretType>& inArr, 
		vector<porthosSecretType>& outArr,
		int32_t sf,
		bool doTruncation)
{
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	log_print("EzPCFunctionalities : Starting Relu2 ... ");

//#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
//#endif
	funcRELUMPC(inArr, outArr, size);
	if (doTruncation && ((partyNum==PARTY_A) || (partyNum==PARTY_B))){
		funcTruncate2PC(outArr, sf, size);
	}
//#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeRelu += tt;
	commObject.dataRelu[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataRelu[1] += (commObject.totalDataReceived - bytesReceived);
//#endif

    commObject.timeTruncation += tt;
    commObject.dataTruncation[0] += (commObject.totalDataSent - bytesSent);
    commObject.dataTruncation[1] += (commObject.totalDataReceived - bytesReceived);
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage relu: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();

}

//////////////////////////////////////////////////
// MaxPool
/////////////////////////////////////////////////

void MaxPool(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C,
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight,
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<porthosSecretType> > > >& inArr,
		vector< vector< vector< vector<porthosSecretType> > > >& outArr)
{
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
    
	log_print("EzPCFunctionalities : Starting MaxPool44 ... ");

//#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
//#endif

	int rows = N*H*W*C;
	int cols = ksizeH*ksizeW;

	vector<porthosSecretType> reInpArr(rows*cols, 0);
	vector<porthosSecretType> maxi(rows, 0);
	vector<porthosSecretType> maxiIdx(rows, 0);

	int rowIdx = 0;
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			int32_t leftTopCornerH = -zPadHLeft;
			int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
			while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
				int32_t leftTopCornerW = -zPadWLeft;
				int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
				while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

					for(int fh=0;fh<ksizeH;fh++){
						for(int fw=0;fw<ksizeW;fw++){
							int32_t colIdx = fh*ksizeW + fw;
							int32_t finalIdx = rowIdx*(ksizeH*ksizeW) + colIdx;

							int32_t curPosH = leftTopCornerH + fh;
							int32_t curPosW = leftTopCornerW + fw;

							porthosSecretType temp = 0;
							if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
								temp = 0;
							}
							else{
								temp = inArr[n][curPosH][curPosW][c];
							}
							reInpArr[finalIdx] = temp;
						}
					}

					rowIdx += 1;
					leftTopCornerW = leftTopCornerW + strideW;
				}

				leftTopCornerH = leftTopCornerH + strideH;
			}
		}
	}

	funcMaxMPC(reInpArr, maxi, maxiIdx, rows, cols);
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					int iidx = n*C*H*W + c*H*W + h*W + w;
					outArr[n][h][w][c] = maxi[iidx];
				}
			}
		}
	}
	
//#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeMaxpool += tt;
	commObject.dataMaxPool[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataMaxPool[1] += (commObject.totalDataReceived - bytesReceived);
//#endif
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage maxpul: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}


//////////////////////////////////////////////////
// AvgPool
/////////////////////////////////////////////////

void AvgPool(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C,
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight,
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<porthosSecretType> > > >& inArr,
		vector< vector< vector< vector<porthosSecretType> > > >& outArr)
{
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	log_print("EzPCFunctionalities : Starting AvgPool44 ... ");
//\#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
//#endif

	int rows = N*H*W*C;
	vector<porthosSecretType> filterAvg(rows, 0);

	int rowIdx = 0;
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			int32_t leftTopCornerH = -zPadHLeft;
			int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
			while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
				int32_t leftTopCornerW = -zPadWLeft;
				int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
				while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

					porthosSecretType curFilterSum = 0;
					for(int fh=0;fh<ksizeH;fh++){
						for(int fw=0;fw<ksizeW;fw++){
							int32_t curPosH = leftTopCornerH + fh;
							int32_t curPosW = leftTopCornerW + fw;

							porthosSecretType temp = 0;
							if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
								temp = 0;
							}
							else{
								temp = inArr[n][curPosH][curPosW][c];
							}

							curFilterSum += temp;
						}
					}

					//IMP NOTE : The local division should always be signed division.
					//TODO : For now doing local truncation : but this will introduce error
					if (partyNum == PARTY_A)
						filterAvg[rowIdx] = static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(curFilterSum))/(ksizeH*ksizeW));

					if (partyNum == PARTY_B)
						filterAvg[rowIdx] = -static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(-curFilterSum))/(ksizeH*ksizeW));

					rowIdx += 1;
					leftTopCornerW = leftTopCornerW + strideW;
				}

				leftTopCornerH = leftTopCornerH + strideH;
			}
		}
	}

	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					int iidx = n*C*H*W + c*H*W + h*W + w;
					outArr[n][h][w][c] = filterAvg[iidx];
				}
			}
		}
	}
	
//#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeAvgPool += tt;
	commObject.dataAvgPool[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataAvgPool[1] += (commObject.totalDataReceived - bytesReceived);
//#endif
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage avgpool: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}

void ElemWiseSecretSharedVectorMult(int32_t size, 
		vector < porthosSecretType > & arr1, 
		vector < porthosSecretType > & arr2, 
		vector < porthosSecretType > & outputArr)
{
    
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	log_print("EzPCFunctionalities : Starting ElemWiseSecretMult ... ");
//#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
//#endif

	funcDotProductMPC(arr1, arr2, outputArr, size, 0, false);

//#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeBN += tt;
	commObject.dataBN[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataBN[1] += (commObject.totalDataReceived - bytesReceived);
//#endif
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage elemwisesecret: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    //peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    //CaptureCpuUsage();
}

void ElemWiseActModelVectorMult(int32_t size, vector<porthosSecretType>& arr1, vector<porthosSecretType>& arr2, vector<porthosSecretType>& outputArr)
{
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	ElemWiseSecretSharedVectorMult(size, arr1, arr2, outputArr);
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage elemwiseact: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
   // peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    //CaptureCpuUsage();
}

void ElemWiseVectorPublicDiv(int32_t size, 
	vector<porthosSecretType>& arr1, 
	int32_t divisor,
	vector<porthosSecretType>& outputArr)
{
    
	//Not being used in our networks right now
	assert(false);
}

void ScaleUp(int32_t s1, vector<porthosSecretType>& arr, int32_t sf)
{
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	for(int i=0;i<s1;i++){
		arr[i] = arr[i] << sf;
	}
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
    cout<<"cpu uage elemwisevector: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}

void ScaleDown(int32_t s1, vector<porthosSecretType>& arr, int32_t sf)
{
    
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
	assert(FLOAT_PRECISION == sf && "Please make FLOAT_PRECISION same as sf used in the network and recompile.");
	if ((partyNum==PARTY_A) || (partyNum==PARTY_B))
		funcTruncate2PC(arr, sf, s1);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
 duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    auto tt = time_span.count();
    commObject.timeTruncation += tt;
    commObject.dataTruncation[0] += (commObject.totalDataSent - bytesSent);
    commObject.dataTruncation[1] += (commObject.totalDataReceived - bytesReceived);
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage scaledown: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}

void Conv2DWrapper(int32_t N, int32_t H, int32_t W, int32_t CI, 
				int32_t FH, int32_t FW, int32_t CO, 
				int32_t zPadHLeft, int32_t zPadHRight, 
				int32_t zPadWLeft, int32_t zPadWRight, 
				int32_t strideH, int32_t strideW, 
				vector< vector< vector< vector<porthosSecretType> > > >& inputArr, 
				vector< vector< vector< vector<porthosSecretType> > > >& filterArr, 
				vector< vector< vector< vector<porthosSecretType> > > >& outArr)
{
    
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
#ifdef CONV_OPTI
	if ((FH>=5) || (FW>=5)){
		funcConv2DCSF(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, 0, outArr);
	}
	else{
		Conv2D(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr);
	}
#else
	Conv2D(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr);
#endif
high_resolution_clock::time_point t2 = high_resolution_clock::now();	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeConv += tt;
	commObject.dataConv[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataConv[1] += (commObject.totalDataReceived - bytesReceived);
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage cpnv2dwrapper: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}

void Conv2DGroupWrapper(int32_t N, int32_t H, int32_t W, int32_t CI,
				int32_t FH, int32_t FW, int32_t CO,
				int32_t zPadHLeft, int32_t zPadHRight,
				int32_t zPadWLeft, int32_t zPadWRight,
				int32_t strideH, int32_t strideW, int32_t G,
				vector< vector< vector< vector<porthosSecretType> > > >& inputArr,
				vector< vector< vector< vector<porthosSecretType> > > >& filterArr,
				vector< vector< vector< vector<porthosSecretType> > > >& outArr)
{
    
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	Conv2DGroup(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, G, inputArr, filterArr, outArr);
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage conv2dgroupwrapper: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}

void Conv3DWrapper(int32_t N, int32_t D, int32_t H, int32_t W, int32_t CI,
                        int32_t FD, int32_t FH, int32_t FW, int32_t CO,
                        int32_t zPadDLeft, int32_t zPadDRight,
                        int32_t zPadHLeft, int32_t zPadHRight,
                        int32_t zPadWLeft, int32_t zPadWRight,
                        int32_t strideD, int32_t strideH, int32_t strideW,
                        vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr,
                        vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr,
                        int32_t consSF,
                        vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr)
{
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	funcConv3DMPC(N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, inputArr, filterArr, consSF, outArr);
    
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage conv3dwrapper: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}

void ConvTranspose3DWrapper(int32_t N, int32_t DPrime, int32_t HPrime, int32_t WPrime, int32_t CI,
                                int32_t FD, int32_t FH, int32_t FW, int32_t CO,
                                int32_t D, int32_t H, int32_t W,
                                int32_t zPadTrDLeft, int32_t zPadTrDRight,
                                int32_t zPadTrHLeft, int32_t zPadTrHRight,
                                int32_t zPadTrWLeft, int32_t zPadTrWRight,
                                int32_t strideD, int32_t strideH, int32_t strideW,
                                vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr,
                                vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr,
                                int32_t consSF,
                                vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr)
{
    // Start timing
    auto startWallTime = std::chrono::high_resolution_clock::now();
    long startCpuTime = GetCpuTime();
    
	ConvTranspose3DCSFMPC(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, inputArr, filterArr, consSF, outArr);
    // Stop timing
    auto endWallTime = std::chrono::high_resolution_clock::now();
    long endCpuTime = GetCpuTime();

    // Calculate elapsed times
    std::chrono::duration<double> elapsedWallTime = endWallTime - startWallTime;
    double elapsedCpuTime = static_cast<double>(endCpuTime - startCpuTime) / sysconf(_SC_CLK_TCK);

    // Calculate and store CPU usage for this segment
    double cpuUsage = (elapsedCpuTime / (elapsedWallTime.count() * numCores)) * 100.0;
    segmentCpuUsages.push_back(cpuUsage);
cout<<"cpu uage convtranspose: "<<cpuUsage<<endl;
    // Update peak CPU usage if the current usage is higher
    peakCpuUsage = std::max(peakCpuUsage, cpuUsage);

    CaptureCpuUsage();
}

void ClearMemSecret1(int64_t s1, vector< porthosSecretType >& arr){
	arr = vector< porthosSecretType >();
}

void ClearMemSecret2(int64_t s1, int64_t s2, vector< vector< porthosSecretType > >& arr){
	arr = vector< vector< porthosSecretType > >();
}

void ClearMemSecret3(int64_t s1, int64_t s2, int64_t s3, vector< vector< vector< porthosSecretType > > >& arr){
	arr = vector< vector< vector< porthosSecretType > > >();

}

void ClearMemSecret4(int64_t s1, int64_t s2, int64_t s3, int64_t s4, vector< vector< vector< vector< porthosSecretType > > > >& arr){
	arr = vector< vector< vector< vector< porthosSecretType > > > >();
}

void ClearMemSecret5(int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, vector< vector< vector< vector< vector< porthosSecretType > > > > >& arr){
	arr = vector< vector< vector< vector< vector< porthosSecretType > > > > >();
}

void ClearMemPublic(int64_t x)
{
	return;
}

void ClearMemPublic1(int64_t s1, vector< int32_t >& arr){
	arr = vector< int32_t >();
}

void ClearMemPublic2(int64_t s1, int64_t s2, vector< vector< int32_t > >& arr){
	arr = vector< vector< int32_t > >();
}

void ClearMemPublic3(int64_t s1, int64_t s2, int64_t s3, vector< vector< vector< int32_t > > >& arr){
	arr = vector< vector< vector< int32_t > > >();

}

void ClearMemPublic4(int64_t s1, int64_t s2, int64_t s3, int64_t s4, vector< vector< vector< vector< int32_t > > > >& arr){
	arr = vector< vector< vector< vector< int32_t > > > >();
}

void ClearMemPublic5(int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, vector< vector< vector< vector< vector< int32_t > > > > >& arr){
	arr = vector< vector< vector< vector< vector< int32_t > > > > >();
}

void Floor(int32_t size, vector<porthosSecretType>& inArr, vector<porthosSecretType>& outArr, int32_t sf)
{
	//Not being used in any network right now
	assert(false);
}


std::chrono::high_resolution_clock::time_point startWallTime;
auto start = std::chrono::high_resolution_clock::now();
long startCpuTime;


void StartComputation(){
    
    std::cout << "Computation started. CPU and wall clocks initialized." << std::endl;
    std::cout << "Number of CPU cores: " << numCores << std::endl;

    // Start measuring CPU time and wall time
    startWallTime = std::chrono::high_resolution_clock::now();
    startCpuTime = GetCpuTime();

    //CaptureCpuUsage();
    
	cout<<"Reached start of computation. Syncronizing across parties..."<<endl;
	synchronize(2000000);
	cout<<"Syncronized - now starting actual execution at "<<getCurrentTime()<<endl;
	start_m();
}

void EndComputation(){
    
	end_m();
    double totalCpuUsage = std::accumulate(segmentCpuUsages.begin(), segmentCpuUsages.end(), 0.0);
    double averageCpuUsage = totalCpuUsage / segmentCpuUsages.size();

    std::cout << "Average CPU usage across all segments: " << averageCpuUsage << " %" << std::endl;

    std::cout << "Peak CPU usage during computation: " << peakCpuUsage << " %" << std::endl;
}

