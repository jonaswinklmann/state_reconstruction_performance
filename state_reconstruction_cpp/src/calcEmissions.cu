__global__ void calcEmissions(const double *inputImage, int *xMin, int *xMax,
    int *yMin, int *yMax, int fullWidth, int partWidth, const double *proj, double *emissions)
{
    int id = threadIdx.x;
    double sum = 0;
    for(int x = xMin[id]; x < xMax[id]; x++)
    {
        for(int y = yMin[id]; y < yMax[id]; y++)
        {
            sum += *(inputImage + x * fullWidth + y) * proj[x * partWidth + y];
        }
    }
    emissions[id] = sum;
}