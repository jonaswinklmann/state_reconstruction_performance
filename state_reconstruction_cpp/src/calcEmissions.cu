#include "calcEmissions.hpp"

#include <chrono>
#include <cuda.h>
#include <fstream>
#include <pybind11/numpy.h>
#include <vector>

#include "data_types.hpp"

#define threadsPerLocalImageDim 8

#define cudaCheck(cmd) {cudaError_t err = cmd; if (err != cudaSuccess) { \
    std::fstream log; \
    log.open("error.log", std::fstream::in | std::fstream::out | std::fstream::app); \
    log << "Failed: Cuda error " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(err) << "'\n"; \
    log.close(); \
    exit(EXIT_FAILURE);}}

__global__ void calcEmissions(const double *inputImage, int *xMin, int *xDist,
    int *yMin, int *yDist, int fullWidth, const double *proj, int *projOffset, double *emissions)
{
    unsigned int imageIndex = blockIdx.x;
    unsigned int workerIndex = threadIdx.x;
    unsigned int workerIndexX = workerIndex / threadsPerLocalImageDim;
    unsigned int workerIndexY = workerIndex % threadsPerLocalImageDim;
    double sum = 0;
    int xMinT = xMin[imageIndex];
    int xDistT = xDist[imageIndex];
    int yMinT = yMin[imageIndex];
    int yDistT = yDist[imageIndex];

    // Calculate emissions for sublocal images
    for(int x = xDistT / threadsPerLocalImageDim * workerIndexX + min(xDistT % threadsPerLocalImageDim, workerIndexX); 
        x < xDistT / threadsPerLocalImageDim * (workerIndexX + 1) + min(xDistT % threadsPerLocalImageDim, workerIndexX + 1); x++)
    {
        for(int y = yDistT / threadsPerLocalImageDim * workerIndexY + min(yDistT % threadsPerLocalImageDim, workerIndexY); 
            y < yDistT / threadsPerLocalImageDim * (workerIndexY + 1) + min(yDistT % threadsPerLocalImageDim, workerIndexY + 1); y++)
        {
            sum += inputImage[(xMinT + x) * fullWidth + yMinT + y] * proj[projOffset[imageIndex] + x * yDistT + y];
        }
    }
    emissions[workerIndex * gridDim.x + imageIndex] = sum;

    // Sum emissions within local images
    unsigned int otherIndex = (workerIndex + 1) * gridDim.x + imageIndex;
    for(unsigned int mask = 2; mask <= threadsPerLocalImageDim * threadsPerLocalImageDim; mask *= 2)
    {
        __syncthreads();
        if(workerIndex % mask == 0)
        {
            emissions[workerIndex * gridDim.x + imageIndex] += emissions[otherIndex];
            otherIndex += mask / 2 * gridDim.x;
        }
    }
}

EmissionCalculatorCUDA::~EmissionCalculatorCUDA()
{
    cudaCheck(cudaFree(this->fullImageDevice));
    cudaCheck(cudaFree(this->projDevice));

    cudaCheck(cudaFree(this->xMin));
    cudaCheck(cudaFree(this->xDist));
    cudaCheck(cudaFree(this->yMin));
    cudaCheck(cudaFree(this->yDist));
    cudaCheck(cudaFree(this->projOffset));
    cudaCheck(cudaFree(this->emissions));
    free(this->xMinHost);
    free(this->xDistHost);
    free(this->yMinHost);
    free(this->yDistHost);
    free(this->projOffsetHost);
    free(this->emissionsHost);
}

void EmissionCalculatorCUDA::initGPUEnvironment()
{
   cudaCheck(cudaFree(0));
}

std::vector<double> EmissionCalculatorCUDA::calcEmissionsGPU(std::vector<Image>& localImages, int psfSupersample)
{
    size_t imageCount = localImages.size();

    if(!this->emissionsHost)
    {
        this->allocatePerImageBuffers((int)imageCount);
    }

    for(size_t i = 0; i < imageCount; i++)
    {
        int xidx = (localImages[i].dx + psfSupersample) % psfSupersample;
        int yidx = (localImages[i].dy + psfSupersample) % psfSupersample;

        this->projOffsetHost[i] = (xidx * (int)(this->projShape[1]) + yidx) * (int)(this->projShape[2]);
        this->xMinHost[i] = localImages[i].X_min;
        this->xDistHost[i] = localImages[i].X_max - localImages[i].X_min + 1;
        this->yMinHost[i] = localImages[i].Y_min;
        this->yDistHost[i] = localImages[i].Y_max - localImages[i].Y_min + 1;
    }

    // Initialize CUDA if not done already
    cudaCheck(cudaFree(0));

    // Transfer data from host to device memory
    cudaCheck(cudaMemcpy(this->xMin, this->xMinHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->xDist, this->xDistHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->yMin, this->yMinHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->yDist, this->yDistHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->projOffset, this->projOffsetHost, sizeof(int) * imageCount, cudaMemcpyHostToDevice));

    calcEmissions<<<imageCount, threadsPerLocalImageDim * threadsPerLocalImageDim>>>(
        this->fullImageDevice, this->xMin, this->xDist, this->yMin, this->yDist, this->imageCols, this->projDevice, this->projOffset, this->emissions);
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(this->emissionsHost, this->emissions, sizeof(double) * imageCount, cudaMemcpyDeviceToHost));

    std::vector<double> emissionsRet(this->emissionsHost, this->emissionsHost + imageCount);
    return emissionsRet;
}

void EmissionCalculatorCUDA::loadImage(const double *image, int imageCols, int imageRows)
{
    if(this->fullImageDevice)
    {
        cudaCheck(cudaFree(this->fullImageDevice));
    }

    cudaCheck(cudaMalloc((void**)&this->fullImageDevice, sizeof(double) * imageRows * imageCols));
    cudaCheck(cudaMemcpy(this->fullImageDevice, image, sizeof(double) * imageRows * imageCols, cudaMemcpyHostToDevice));

    this->imageCols = imageCols;
}

void EmissionCalculatorCUDA::loadProj(py::object& prjgen)
{
    bool projCacheBuilt = prjgen.attr("proj_cache_built").cast<bool>();
    if(!projCacheBuilt)
    {
        prjgen.attr("setup_cache")();
    }
    py::array_t<double> projs = prjgen.attr("proj_cache").cast<py::array_t<double>>();

    const ssize_t *shape = projs.shape();
    projs = projs.reshape(std::vector<int>({(int)(shape[0]), (int)(shape[1]), -1}));
    this->projShape = projs.shape();

    if(this->projDevice)
    {
        cudaCheck(cudaFree(this->projDevice));
    }

    cudaCheck(cudaMalloc((void**)&this->projDevice, sizeof(double) * projs.size()));
    cudaCheck(cudaMemcpy(this->projDevice, projs.data(), sizeof(double) * projs.size(), cudaMemcpyHostToDevice));
}

void EmissionCalculatorCUDA::allocatePerImageBuffers(int imageCount)
{
    if(this->emissionsHost)
    {
        cudaCheck(cudaFree(this->xMin));
        cudaCheck(cudaFree(this->xDist));
        cudaCheck(cudaFree(this->yMin));
        cudaCheck(cudaFree(this->yDist));
        cudaCheck(cudaFree(this->projOffset));
        cudaCheck(cudaFree(this->emissions));
        free(this->xMinHost);
        free(this->xDistHost);
        free(this->yMinHost);
        free(this->yDistHost);
        free(this->projOffsetHost);
        free(this->emissionsHost);
    }

    // Allocate host memory
    this->emissionsHost = static_cast<double*>(malloc(sizeof(double) * imageCount));
    this->xMinHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    this->xDistHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    this->yMinHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    this->yDistHost = static_cast<int*>(malloc(sizeof(int) * imageCount));
    this->projOffsetHost = static_cast<int*>(malloc(sizeof(int) * imageCount));

    // Allocate device memory
    cudaCheck(cudaMalloc((void**)&this->xMin, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&this->xDist, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&this->yMin, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&this->yDist, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&this->projOffset, sizeof(int) * imageCount));
    cudaCheck(cudaMalloc((void**)&this->emissions, sizeof(double) * threadsPerLocalImageDim * threadsPerLocalImageDim * imageCount));
}