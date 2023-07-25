#include "calcEmissions.hpp"

#include <cuda.h>
#include <iostream>
#include <vector>

#include "data_types.hpp"

#define threadsPerLocalImageDim 8

#define cudaCheck(cmd) {cudaError_t err = cmd; if (err != cudaSuccess) { std::cout << "Failed: Cuda error " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(err) << "'\n"; exit(EXIT_FAILURE);}}

__global__ void calcEmissions(const double *inputImage, int *xMin, int *xMax,
    int *yMin, int *yMax, int fullWidth, const double *proj, int *projOffset, double *emissions)
{
    unsigned int imageIndex = threadIdx.x;
    unsigned int workerIndex = blockIdx.x;
    unsigned int workerIndexX = workerIndex / threadsPerLocalImageDim;
    unsigned int workerIndexY = workerIndex % threadsPerLocalImageDim;
    double sum = 0;
    int xMinT = xMin[imageIndex];
    int xDist = xMax[imageIndex] - xMinT;
    int yMinT = yMin[imageIndex];
    int yDist = yMax[imageIndex] - yMinT;

    // Calculate emissions for sublocal images
    for(int x = xDist / blockDim.x * workerIndexX; x < xDist / blockDim.x * (workerIndexX + 1); x++)
    {
        for(int y = yDist / blockDim.x * workerIndexY; y < yDist / blockDim.x * (workerIndexY + 1); y++)
        {
            sum += inputImage[(xMinT + x) * fullWidth + yMinT + y] * proj[projOffset[imageIndex] + x * yDist + y];
        }
    }
    emissions[workerIndex * gridDim.x + imageIndex] = sum;

    // Sum emissions within local images
    unsigned int otherIndex = (workerIndex + 1) * gridDim.x + imageIndex;
    for(unsigned int mask = 2; mask <= threadsPerLocalImageDim * threadsPerLocalImageDim; mask <<= 1)
    {
        __syncthreads();
        if(workerIndex % mask == 0)
        {
            emissions[workerIndex * gridDim.x + imageIndex] += emissions[otherIndex];
            otherIndex += mask / 2 * gridDim.x;
        }
    }
}

std::vector<double> calcEmissionsGPU(const double *image, int imageRows, int imageCols, 
    const double *projIn, int projSize, const ssize_t *projShape, std::vector<Image>& localImages, int psfSupersample)
{
    double *inputImage, *proj;
    double *emissions;
    int *xMin, *xMax, *yMin, *yMax, *projOffset;
    size_t imageCount = localImages.size();

    double *emissionsHost = static_cast<double*>(malloc(sizeof(double) * imageCount));
    int *xMinHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    int *xMaxHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    int *yMinHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    int *yMaxHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    int *projOffsetHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    for(size_t i = 0; i < imageCount; i++)
    {
        int xidx = localImages[i].dx % psfSupersample;
        int yidx = localImages[i].dy % psfSupersample;

        projOffsetHost[i] = (xidx * (int)(projShape[1]) + yidx) * (int)(projShape[2]);
        xMinHost[i] = localImages[i].X_min;
        xMaxHost[i] = localImages[i].X_max;
        yMinHost[i] = localImages[i].Y_min;
        yMaxHost[i] = localImages[i].Y_max;
    }

    // Initialize CUDA
    cudaCheck(cudaFree(0));

    // Allocate device memory for a
    cudaCheck(cudaMalloc((void**)&inputImage, sizeof(double) * imageRows * imageCols));
    cudaCheck(cudaMalloc((void**)&proj, sizeof(double) * projSize));
    cudaCheck(cudaMalloc((void**)&xMin, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&xMax, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&yMin, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&yMax, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&projOffset, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&emissions, sizeof(double) * threadsPerLocalImageDim * threadsPerLocalImageDim * imageCount));

    // Transfer data from host to device memory
    cudaCheck(cudaMemcpy(inputImage, image, sizeof(double) * imageRows * imageCols, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(proj, projIn, sizeof(double) * projSize, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(xMin, xMinHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(xMax, xMaxHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(yMin, yMinHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(yMax, yMaxHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(projOffset, projOffsetHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));

    calcEmissions<<<imageCount, threadsPerLocalImageDim * threadsPerLocalImageDim>>>(inputImage, xMin, xMax, yMin, yMax, imageCols, proj, projOffset, emissions);
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(emissionsHost, emissions, sizeof(double) * imageCount, cudaMemcpyDeviceToHost));

    std::vector<double> emissionsRet(emissionsHost, emissionsHost + imageCount);

    cudaCheck(cudaFree(inputImage));
    cudaCheck(cudaFree(proj));
    cudaCheck(cudaFree(xMin));
    cudaCheck(cudaFree(xMax));
    cudaCheck(cudaFree(yMin));
    cudaCheck(cudaFree(yMax));
    cudaCheck(cudaFree(emissions));
    free(emissionsHost);
    return emissionsRet;
}