#ifndef MNIST_LOADER_CUH
#define MNIST_LOADER_CUH

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

struct MNISTData {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

class MNISTLoader {
public:
    MNISTData load(const std::string& images_path, const std::string& labels_path) {
        MNISTData data;
        // Load images
        std::ifstream images_file(images_path, std::ios::binary);
        if (!images_file.is_open()) {
            throw std::runtime_error("Failed to open images file: " + images_path);
        }
        // Skip header
        images_file.ignore(16);
        // Read image data
        while (!images_file.eof()) {
            std::vector<float> image(28 * 28); // MNIST images are 28x28
            images_file.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(float));
            if (images_file.gcount() == image.size() * sizeof(float)) {
                data.images.push_back(image);
            }
        }
        images_file.close();

        // Load labels
        std::ifstream labels_file(labels_path, std::ios::binary);
        if (!labels_file.is_open()) {
            throw std::runtime_error("Failed to open labels file: " + labels_path);
        }
        // Skip header
        labels_file.ignore(8);
        // Read label data
        while (!labels_file.eof()) {
            char label;
            labels_file.read(&label, 1);
            if (labels_file.gcount() == 1) {
                data.labels.push_back(static_cast<int>(label));
            }
        }
        labels_file.close();

        return data;
    }

    std::vector<std::vector<std::vector<float>>> create_patches(const std::vector<std::vector<float>>& images, int patch_size) {
        std::vector<std::vector<std::vector<float>>> patches;
        for (const auto& image : images) {
            std::vector<std::vector<float>> image_patches;
            for (int i = 0; i < 28; i += patch_size) {
                for (int j = 0; j < 28; j += patch_size) {
                    std::vector<float> patch;
                    for (int pi = 0; pi < patch_size; ++pi) {
                        for (int pj = 0; pj < patch_size; ++pj) {
                            int idx = (i + pi) * 28 + (j + pj);
                            if (idx < image.size()) {
                                patch.push_back(image[idx]);
                            }
                        }
                    }
                    image_patches.push_back(patch);
                }
            }
            patches.push_back(image_patches);
        }
        return patches;
    }
};

#endif
