#include <cmath>
#include <fourier_encoding.h>


const int input_size = encodedInputSize;

void encodePosition(float x, float y, float* out) {
    int i = 0;
    for (int l = 0; l < L; l++) {
        float freq = powf(2.0f, l) * 3.1415926f;

        out[i++] = sinf(freq * x);
        out[i++] = cosf(freq * x);
        out[i++] = sinf(freq * y);
        out[i++] = cosf(freq * y);
    }
}
