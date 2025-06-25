// optimizer.cuh
#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include <cuda_runtime.h>
#include "common.cuh"

class Optimizer {
public:
    Optimizer(float learning_rate);
    virtual void step(float* params, float* grads, size_t size) = 0;
    float getLearningRate() const { return learning_rate; }

protected:
    float learning_rate;
};

class SGD : public Optimizer {
public:
    SGD(float learning_rate);
    void step(float* params, float* grads, size_t size) override;
};

class Adam : public Optimizer {
public:
    Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    void step(float* params, float* grads, size_t size) override;

private:
    float beta1, beta2, epsilon;
    float* m; // First moment vector
    float* v; // Second moment vector
    size_t timestep;
};

#endif // OPTIMIZER_CUH