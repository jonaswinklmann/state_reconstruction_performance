#include <stddef.h>

struct Image
{
    const double *image;
    size_t offset, outerStride, innerStride;
    // Rounded PSF center coordinates.
    int X_int, Y_int;
    // Rounded PSF rectangle corners.
    int X_min, X_max, Y_min, Y_max;
    // Subpixel shifts.
    int dx, dy;
};