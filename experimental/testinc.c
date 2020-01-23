#include <stdint.h>

#define MIN_GREEN       128
#define MIN_SATURATION  128
#define H_MIN           128
#define H_MAX           213

#define X_RES           320
#define Y_RES           240

#define DST_SIZE       X_RES*Y_RES

/**
 * Given an image in BGR format, one byte per channel, in src, and a pre-cleared
 * array in dst, generate a single-channel mask such that pixels meeting the
 * following conditions are white in dst:
 *    G >= MIN_GREEN
 *    G > B > R
 *    (G - R)*256 / G >= MIN_SATURATION
 *    H_MIN <= (B - R)*256 / (G - R) <= H_MAX
 */
void test_homemade(uint8_t *src, uint8_t *dst) {
    uint8_t *b = src;
    uint8_t *g = src+1;
    uint8_t *r = src+2;
    uint8_t *end_dst = dst + DST_SIZE;
    while (dst < end_dst) {
        if (*g > MIN_GREEN) {
            if ((*g > *b) && (*b > *r)) {
                unsigned int chroma = (*g - *r);
                // Multiply on the other side to avoid a division
                if ((chroma << 8) <= MIN_SATURATION * *g) {
                    unsigned int h = (*b - *r) << 8;
                    // And again we multiply the max rather than divide
                    unsigned int max = H_MAX * chroma;
                    if ((h >= H_MIN) && (h <= max)) {
                        *dst = 255;
                    }
                }
            }
        }
        b+=3;
        g+=3;
        r+=3;
        dst++;
    }
}