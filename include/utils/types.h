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

    // --- Offline indexing parameters (static build phase) ---
    int num_clusters = 2000;
    int max_iterations = 3;
    int max_blocks_per_dimension = 1000;  // nb_build: max inverted blocks retained per concept
    int max_docs_per_block = 500;         // nd_build: max documents retained per block

    // --- Online search hyperparameters (dynamic query phase) ---
    float heap_factor = 0.15f;            // relaxation multiplier for block pruning
    int max_query_terms = 0;              // max highest-weighted query coordinates to evaluate (0 = all)
    int max_search_blocks = 0;            // max inverted blocks to evaluate per query term (0 = all)
    int max_docs_to_visit = 0;            // hard termination cap on exact dot products (0 = unlimited)

    bool log_debug = false;
    bool mem_debug = false;
};