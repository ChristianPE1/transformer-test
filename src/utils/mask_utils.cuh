#ifndef MASK_UTILS_CUH
#define MASK_UTILS_CUH

#include <cuda_runtime.h>
#include <vector>

class MaskUtils {
public:
    static void createPaddingMask(const std::vector<int> &tokens, float *mask, int pad_token);
    static void createLookAheadMask(float *mask, int seq_len);
    static void combineDecoderMasks(const std::vector<int> &tokens, float *combined_mask, int pad_token);
};

#endif // MASK_UTILS_CUH