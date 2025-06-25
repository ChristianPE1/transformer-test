NVCC = nvcc
CXXFLAGS = -std=c++17 -O2
CUDAFLAGS = -arch=sm_70 -rdc=true
INCLUDES = -Iinclude -Isrc

# Source files
DATA_SOURCES = src/data/dataset.cu src/data/vocab.cu src/data/tsv_parser.cu
UTILS_SOURCES = src/utils/matrix.cu
TRANSFORMER_SOURCES = src/transformer/transformer.cu src/transformer/embeddings.cu

ALL_SOURCES = $(DATA_SOURCES) $(UTILS_SOURCES) $(TRANSFORMER_SOURCES)

# Main targets
all: test_simple main_transformer

test_simple: test_simple.cu
    $(NVCC) $(CUDAFLAGS) -std=c++17 $(INCLUDES) $< -o $@

main_transformer: src/main.cu $(ALL_SOURCES)
    $(NVCC) $(CUDAFLAGS) -std=c++17 $(INCLUDES) $^ -o $@

clean:
    rm -f test_simple main_transformer

.PHONY: all clean