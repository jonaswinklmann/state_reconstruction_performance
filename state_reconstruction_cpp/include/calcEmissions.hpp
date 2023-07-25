#include <vector>
#include <stddef.h>

struct Image;
std::vector<double> calcEmissionsGPU(const double *image, int imageRows, int imageCols, 
    const double *projIn, int projSize, const ssize_t *projShape, std::vector<Image>& localImages, int psfSupersample);