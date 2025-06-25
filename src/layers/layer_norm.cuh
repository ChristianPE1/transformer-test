// layer_norm.cuh
#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

class LayerNorm {
public:
    LayerNorm(size_t d_model, double epsilon = 1e-6);
    Matrix forward(const Matrix &input);
    
private:
    size_t d_model;
    double epsilon;
    Matrix gamma;
    Matrix beta;
};

#endif // LAYER_NORM_H