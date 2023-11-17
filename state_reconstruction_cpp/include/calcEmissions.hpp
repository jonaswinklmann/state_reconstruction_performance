#include <vector>
#include <stddef.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Image;
class EmissionCalculatorCUDA
{
private:
    double *fullImageDevice, *projDevice;
    pybind11::ssize_t *projShape;
    int imageCols;

    double *emissions, *emissionsHost;
    int *xMin, *xDist, *yMin, *yDist, *projOffset;
    int *xMinHost, *xDistHost, *yMinHost, *yDistHost, *projOffsetHost;
public:
    EmissionCalculatorCUDA() {};
    static EmissionCalculatorCUDA& getInstance()
    {
        static EmissionCalculatorCUDA instance;
        return instance;
    };
    EmissionCalculatorCUDA(EmissionCalculatorCUDA const&) = delete;
    void operator=(EmissionCalculatorCUDA const&) = delete;
    ~EmissionCalculatorCUDA();
    std::vector<double> calcEmissionsGPU(std::vector<Image>& localImages, int psfSupersample);
    void initGPUEnvironment();
    void loadImage(const double *image, int imageCols, int imageRows);
    void loadProj(py::object& prjgen);
    void allocatePerImageBuffers(int imageCount);
};