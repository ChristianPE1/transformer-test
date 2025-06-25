// filepath: cuda-transformer/cuda-transformer/src/training/trainer.cuh
#ifndef TRAINER_CUH
#define TRAINER_CUH

#include <vector>
#include "transformer/transformer.cuh"
#include "training/loss.cuh"
#include "training/optimizer.cuh"

class Trainer {
public:
    Trainer(Transformer& model, Optimizer& optimizer, Loss& loss_fn, int batch_size, int epochs);
    void train(const std::vector<std::vector<int>>& source_batches, const std::vector<std::vector<int>>& target_batches);
    void evaluate(const std::vector<std::vector<int>>& source_batches, const std::vector<std::vector<int>>& target_batches);

private:
    Transformer& model;
    Optimizer& optimizer;
    Loss& loss_fn;
    int batch_size;
    int epochs;

    void forward_and_backward(const std::vector<int>& source, const std::vector<int>& target);
};

#endif // TRAINER_CUH