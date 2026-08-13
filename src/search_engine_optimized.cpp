#include "search_engine_optimized.h"
#include <iostream>
#include <queue>
#include <algorithm>
#include <numeric>

std::vector<std::pair<float, int>> SearchEngineOptimized::search(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& train,
    const std::vector<std::vector<InvertedBlock>>& inverted_index,
    const std::vector<Eigen::VectorXf>& summary_vectors,
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& query_matrix,
    int q_idx,
    int k,
    const struct ExecConfig& config
) {
    int max_docs_to_visit = config.max_docs_to_visit;
    float heap_factor = config.heap_factor;
    int max_query_terms = config.max_query_terms;
    int max_search_blocks = config.max_search_blocks;
    int num_clusters = summary_vectors.size();
    int n_docs = train.rows();
    int n_dims = train.cols();

    std::vector<std::pair<int, float>> query_terms;

    // gather sparse (concept_id, weight) pairs first so mqt can define the effective query
    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(query_matrix, q_idx); it; ++it) {
        query_terms.push_back({it.index(), it.value()});
    }

    // Sort coordinates descending to push heap thresholds up early
    std::sort(query_terms.begin(), query_terms.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Truncate long tail of sparse query to the top max_query_terms coordinates
    if (max_query_terms > 0 && static_cast<int>(query_terms.size()) > max_query_terms) {
        query_terms.resize(max_query_terms);
    }

    // Build dense query from the effective (possibly truncated) term set only.
    std::vector<float> query_dense(n_dims, 0.0f);
    for (const auto& [concept_id, q_weight] : query_terms) {
        query_dense[concept_id] = q_weight;
    }

    std::vector<float> cluster_ub(num_clusters, 0.0f);
    for (int c = 0; c < num_clusters; ++c) {
        for (const auto& [concept_id, q_weight] : query_terms) {
            cluster_ub[c] += q_weight * summary_vectors[c][concept_id];
        }
    }

    // Format: (score, doc_id). The minimum score of our Top-K sits on top as our threshold T.
    std::priority_queue<std::pair<float, int>, 
                        std::vector<std::pair<float, int>>, 
                        std::greater<std::pair<float, int>>> min_heap;

    std::vector<bool> visited(n_docs, false);

    int num_of_docs_visited = 0;
    int blocks_entered = 0;
    int blocks_skipped = 0;
    int docs_popped = 0;
    bool stop_search = false;

    for (const auto& [concept_id, q_weight] : query_terms) {
        if (stop_search) break;
        if (concept_id >= inverted_index.size()) continue;

        const auto& blocks = inverted_index[concept_id];

        // The index is already built with the most promising clusters first for each concept.
        // Avoid per-query heap allocations and sorts by scanning blocks in stored order.
        int blocks_evaluated_for_term = 0;
        for (const auto& block : blocks) {
            // Truncate: only evaluate up to max_search_blocks per query term
            if (max_search_blocks > 0 && blocks_evaluated_for_term >= max_search_blocks) break;
            ++blocks_evaluated_for_term;
            int c_id = block.cluster_id;
            float ub = cluster_ub[c_id];

            // get the WORST score in the current priority queue
            float threshold = min_heap.empty() ? 0.0f : min_heap.top().first;

            // If the upper bound is below the threshold, skip the entire block.
            if (ub < threshold * heap_factor) {
                ++blocks_skipped;
            } else {
                ++blocks_entered;
                for (int doc_id : block.doc_ids) {
                    if (!visited[doc_id]) {
                        visited[doc_id] = true;
                        ++num_of_docs_visited;

                        float exact_score = 0.0f;
                        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator doc_it(train, doc_id); doc_it; ++doc_it) {
                            exact_score += doc_it.value() * query_dense[doc_it.index()];
                        }

                        if (min_heap.size() < k) {
                            min_heap.push({exact_score, doc_id});
                        } else if (exact_score > min_heap.top().first) {
                            min_heap.pop();
                            min_heap.push({exact_score, doc_id});
                            ++docs_popped;
                        }

                        if (max_docs_to_visit > 0 && num_of_docs_visited >= max_docs_to_visit) {
                            stop_search = true;
                            break;
                        }
                    }
                    if (stop_search) break;
                }
                if (stop_search) break;
            }
            if (stop_search) break;
        }
    }

    // Accumulate into the object's totals; caller will print averages after all queries
    total_blocks_entered += blocks_entered;
    total_blocks_skipped += blocks_skipped;
    total_docs_examined += num_of_docs_visited;
    total_docs_popped += docs_popped;
    ++num_queries_run;

    std::vector<std::pair<float, int>> top_k;
    while (!min_heap.empty()) {
        top_k.push_back(min_heap.top());
        min_heap.pop();
    }
    std::reverse(top_k.begin(), top_k.end());

    return top_k;
}

void SearchEngineOptimized::printAvgDebugStats() const {
    if (num_queries_run == 0) {
        std::cout << "[SEARCH_DEBUG_AVG] No queries run.\n";
        return;
    }
    double avg_blocks_entered = static_cast<double>(total_blocks_entered) / static_cast<double>(num_queries_run);
    double avg_blocks_skipped = static_cast<double>(total_blocks_skipped) / static_cast<double>(num_queries_run);
    double avg_docs_examined = static_cast<double>(total_docs_examined) / static_cast<double>(num_queries_run);
    double avg_docs_popped = static_cast<double>(total_docs_popped) / static_cast<double>(num_queries_run);

    std::cout << "[SEARCH_DEBUG_AVG] avg_blocks_entered=" << avg_blocks_entered
              << " avg_blocks_skipped=" << avg_blocks_skipped
              << " avg_docs_examined=" << avg_docs_examined
              << " avg_docs_popped=" << avg_docs_popped
              << "\n";
}

void SearchEngineOptimized::getAvgDebugStats(double& avg_blocks_entered, double& avg_blocks_skipped, double& avg_docs_examined, double& avg_docs_popped) const {
    if (num_queries_run == 0) {
        avg_blocks_entered = 0.0;
        avg_blocks_skipped = 0.0;
        avg_docs_examined = 0.0;
        avg_docs_popped = 0.0;
        return;
    }
    avg_blocks_entered = static_cast<double>(total_blocks_entered) / static_cast<double>(num_queries_run);
    avg_blocks_skipped = static_cast<double>(total_blocks_skipped) / static_cast<double>(num_queries_run);
    avg_docs_examined = static_cast<double>(total_docs_examined) / static_cast<double>(num_queries_run);
    avg_docs_popped = static_cast<double>(total_docs_popped) / static_cast<double>(num_queries_run);
}