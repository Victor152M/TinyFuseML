#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <image_writer.h>

void SaveRgbImage(const unsigned char* data, int width, int height, const char* filename) {
    stbi_write_png(filename, width, height, 3, data, width * 3);
}
