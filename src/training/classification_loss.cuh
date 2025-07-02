#ifndef CLASSIFICATION_LOSS_CUH
#define CLASSIFICATION_LOSS_CUH

#include "../utils/matrix.cuh"

class CrossEntropyLoss {
public:
    static float compute_loss(const Matrix& predictions, const std::vector<int>& labels);
    static Matrix compute_gradients(const Matrix& predictions, const std::vector<int>& labels);
};

#endif // CLASSIFICATION_LOSS_CUH
