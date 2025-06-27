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
    float train(const std::vector<std::vector<int>>& source_batches, const std::vector<std::vector<int>>& target_batches);  // NOW RETURNS LOSS
    void evaluate(const std::vector<std::vector<int>>& source_batches, const std::vector<std::vector<int>>& target_batches);
    void setVerbose(bool verbose) { this->verbose = verbose; }  // NEW METHOD

private:
    Transformer& model;
    Optimizer& optimizer;
    Loss& loss_fn;
    int batch_size;
    int epochs;
    int global_step; // Track training steps for learning rate scheduling
    bool verbose = true;  // NEW MEMBER

    void forward_and_backward(const std::vector<int>& source, const std::vector<int>& target);
    float calculateLearningRate(int step, float current_loss);
    double calculateEOSPenalty(const Matrix& predictions, const std::vector<int>& target_sequence);
};

#endif // TRAINER_CUH