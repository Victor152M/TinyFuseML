#pragma once

// Set L to control frequency depth
constexpr int L = 10;
constexpr int encodedInputSize = 2 * L * 2;

/**
 * Encodes 2D position (x, y) using sine and cosine functions
 * with exponentially increasing frequencies.
 *
 * @param x     Normalized X position in [0, 1]
 * @param y     Normalized Y position in [0, 1]
 * @param out   Output array of size encoded_input_size
 */
void encodePosition(float x, float y, float* out);