#pragma once
#include <Eigen/Sparse>
#include <vector>

// Forward Index: just an alias for easier reading
using ForwardIndex = Eigen::SparseMatrix<float, Eigen::RowMajor>;

struct Block {
    std::vector<int> doc_ids;       // Indices of documents in this block
    Eigen::VectorXf summary;        // The "Sketch" (Max-weights for pruning)
};

/**
 * @brief Represents a block of documents within a single concept's inverted list.
 * All documents in this block belong to the same cluster.
 */
struct InvertedBlock {
    int cluster_id;
    std::vector<int> doc_ids;
    std::vector<float> weights;
};

struct ClusterResult {
    std::vector<int> assignments; // Which cluster each doc belongs to
    std::vector<Eigen::VectorXf> centroids; // The dense/sparse centers
};

struct ExecConfig {
    std::string config_file = "exec_config.json";
    int num_clusters = 200;
    int max_iterations = 3;
    int max_blocks_per_dimension = 0;
    int max_docs_per_block = 0;
    int max_docs_to_visit = 0;
    float heap_factor = 0.60f;
    bool log_debug = false;
    bool mem_debug = false;
};