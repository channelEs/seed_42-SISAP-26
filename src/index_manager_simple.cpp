#include "index_manager_simple.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <unistd.h>
#include <cstdint>

static double estimateIndexMemoryGBytes(int n_dims, int num_clusters, const struct ExecConfig& config) {
    uint64_t max_docs = (config.max_docs_per_block > 0) ? static_cast<uint64_t>(config.max_docs_per_block) : static_cast<uint64_t>(n_dims);
    uint64_t cluster_count = static_cast<uint64_t>(num_clusters);

    unsigned long long pair_size = static_cast<unsigned long long>(sizeof(std::pair<int, float>));
    unsigned long long estimate_docs = static_cast<unsigned long long>(n_dims) * cluster_count * max_docs;
    unsigned long long docs_bytes = estimate_docs * (pair_size * 2); // account for heap/vector overhead

    unsigned long long cluster_nodes = static_cast<unsigned long long>(n_dims) * cluster_count;
    unsigned long long cluster_overhead = cluster_nodes * 80ULL;

    unsigned long long total = docs_bytes + cluster_overhead;
    return static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
}

std::vector<Eigen::VectorXf> IndexManagerSimple::computeSummaryVectors(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
    const std::vector<int>& assignments,
    int num_clusters
) {
    int n_docs = data.rows();
    int n_dims = data.cols();

    // Initialize dense vectors to store the maximum cluster weight for each concept.
    // This ensures the summary produces a valid upper bound for pruning.
    std::vector<Eigen::VectorXf> summaries(num_clusters, Eigen::VectorXf::Zero(n_dims));

    std::cout << "[INDEXING] Computing Summary Vectors (Sketches) for " << num_clusters << " clusters..." << std::endl;

    for (int i = 0; i < n_docs; ++i) {
        int cluster_id = assignments[i];

        // Use Eigen's InnerIterator to touch only non-zero values (High Efficiency)
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, i); it; ++it) {
            int term_idx = it.index();
            float weight = it.value();
            if (weight > summaries[cluster_id][term_idx]) {
                summaries[cluster_id][term_idx] = weight;
            }
        }
    }

    return summaries;
}

std::vector<std::vector<InvertedBlock>> IndexManagerSimple::buildInvertedIndex(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
    const std::vector<int>& assignments,
    int num_clusters,
    const struct ExecConfig& config,
    const std::vector<Eigen::VectorXf>& summaries
) {
    int n_docs = data.rows();
    int n_dims = data.cols();

    std::cout << "[INDEXING] SIMPLE buildInvertedIndex start: n_docs=" << n_docs
              << " n_dims=" << n_dims
              << " nonzeros=" << data.nonZeros()
              << " max_blocks_per_dimension=" << config.max_blocks_per_dimension
              << " max_docs_per_block=" << config.max_docs_per_block
              << " max_docs_to_visit=" << config.max_docs_to_visit
              << std::endl;

    if (config.mem_debug) {
        const double estimated_index_gbytes = estimateIndexMemoryGBytes(n_dims, num_clusters, config);
        std::cout << "[MEMORY_ESTIMATE] Estimated build memory=" << estimated_index_gbytes
                  << " GB. Proceeding with index build regardless of system limits." << std::endl;
    }

    try {
        // The partitioned index: Concept -> List of Blocks
        std::vector<std::vector<InvertedBlock>> index(n_dims);

        // std::cout << "[INDEXING] Building inverted index from summary vectors (simple approach)..." << std::endl;

        // collect all documents per (concept, cluster) in a single pass over non-zeros
        using DocWeight = std::pair<int, float>;
        std::vector<std::unordered_map<int, std::vector<DocWeight>>> concept_docs(n_dims);
        for (int i = 0; i < n_docs; ++i) {
            int cluster_id = assignments[i];
            for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(data, i); it; ++it) {
                int concept_id = it.index();
                float weight = it.value();
                concept_docs[concept_id][cluster_id].emplace_back(i, weight);
            }
        }

        const int nb = (config.max_blocks_per_dimension > 0) ? config.max_blocks_per_dimension : num_clusters;
        const int nd = (config.max_docs_per_block > 0) ? config.max_docs_per_block : n_docs;

        int processed_concepts = 0;
        for (int concept_id = 0; concept_id < n_dims; ++concept_id) {
            // Build cluster weight list from summaries: cluster -> weight for this concept
            std::vector<std::pair<int, float>> cluster_weights;
            cluster_weights.reserve(num_clusters);
            for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
                float w = summaries[cluster_id][concept_id];
                if (w > 0.0f) cluster_weights.emplace_back(cluster_id, w);
            }
            if (cluster_weights.empty()) continue;
            ++processed_concepts;

            // Keep top-nb clusters by summary weight
            if ((int)cluster_weights.size() > nb) {
                auto nth = cluster_weights.begin() + nb;
                std::nth_element(cluster_weights.begin(), nth, cluster_weights.end(),
                    [](const auto &a, const auto &b){ return a.second > b.second; });
                cluster_weights.erase(nth, cluster_weights.end());
            }
            std::sort(cluster_weights.begin(), cluster_weights.end(), [](const auto &a, const auto &b){ return a.second > b.second; });

            // For each selected cluster, pick top-nd documents by their weight for this concept
            for (auto &cw : cluster_weights) {
                int cluster_id = cw.first;
                auto it_map = concept_docs[concept_id].find(cluster_id);
                if (it_map == concept_docs[concept_id].end()) continue;
                auto docs = std::move(it_map->second);
                if (docs.empty()) continue;

                std::sort(docs.begin(), docs.end(), [](const DocWeight &a, const DocWeight &b){ return a.second > b.second; });
                int keep_docs = static_cast<int>(docs.size());
                if (keep_docs > nd) keep_docs = nd;

                InvertedBlock block;
                block.cluster_id = cluster_id;
                block.doc_ids.reserve(keep_docs);
                block.weights.reserve(keep_docs);
                for (int j = 0; j < keep_docs; ++j) {
                    block.doc_ids.push_back(docs[j].first);
                    block.weights.push_back(docs[j].second);
                }
                index[concept_id].push_back(std::move(block));
            }
        }

        // Profiling: compute totals and averages for blocks and documents
        size_t total_blocks = 0;
        size_t total_docs_in_blocks = 0;
        int empty_dimensions = 0;
        for (int concept_id = 0; concept_id < n_dims; ++concept_id) {
            auto &blocks = index[concept_id];
            if (blocks.empty()) {
                ++empty_dimensions;
                continue;
            }
            total_blocks += blocks.size();
            for (const auto &b : blocks) {
                total_docs_in_blocks += b.doc_ids.size();
            }
        }

        double avg_docs_per_block = 0.0;
        if (total_blocks > 0) avg_docs_per_block = static_cast<double>(total_docs_in_blocks) / static_cast<double>(total_blocks);
        double avg_blocks_per_dimension = static_cast<double>(total_blocks) / static_cast<double>(n_dims);

        std::cout << "[INDEX_PROFILE] total_blocks=" << total_blocks
                  << " total_docs_in_blocks=" << total_docs_in_blocks
                  << " avg_docs_per_block=" << avg_docs_per_block
                  << " avg_blocks_per_dimension=" << avg_blocks_per_dimension
                  << " empty_dimensions=" << empty_dimensions << std::endl;

        return index;
    } catch (const std::bad_alloc& e) {
        std::cerr << "[ERROR] buildInvertedIndex ran out of memory: " << e.what() << std::endl;
        std::cerr << "[ERROR] n_docs=" << n_docs
                  << " n_dims=" << n_dims
                  << " max_blocks_per_dimension=" << config.max_blocks_per_dimension
                  << " max_docs_per_block=" << config.max_docs_per_block
                  << " max_docs_to_visit=" << config.max_docs_to_visit
                  << std::endl;
        throw;
    }
}