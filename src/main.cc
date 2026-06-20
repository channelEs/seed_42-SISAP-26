#include <iostream>
#include <fstream>
#include <sstream>
#include <new>
#include "utils/hdf5_sparse_loader.h"
#include "utils/runtime_helpers.h"
#include "utils/types.h"
#include "utils/metrics.h"
#include "clustering_engine.h"
#include "index_manager.h"
#include "index_manager_simple.h"
#include "index_manager_optimized.h"

#include "search_engine.h"
#include "search_engine_simple.h"
#include "search_engine_optimized.h"

#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <filesystem>
#include <Eigen/Sparse>

static const std::string STATIC_DATASET = "nq";
static const std::string STATIC_TASK = "task3";

int main(int argc, char* argv[]) {
    try {
        std::set_new_handler([]() {
            std::cerr << "[ERROR] Memory allocation failure: std::new_handler triggered.\n";
        });

        // Default fallbacks if flags aren't passed
        std::string dataset = STATIC_DATASET;
        std::string task = STATIC_TASK;
        std::string config_folder = "clusters";
        std::string input_h5_path;
        std::string task_description_path;
        std::filesystem::path output_root = "results";

        std::filesystem::path params_path = std::filesystem::path("config") / config_folder;
        std::vector<std::filesystem::path> config_paths;

        // Parse command line arguments from SISAP only in development mode
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dataset" && i + 1 < argc) {
                dataset = argv[++i];
            } else if (arg == "--task" && i + 1 < argc) {
                task = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                config_folder = argv[++i];
            } else if (arg == "--input" && i + 1 < argc) {
                input_h5_path = argv[++i];
            } else if (arg == "--task-description" && i + 1 < argc) {
                task_description_path = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output_root = std::filesystem::path(argv[++i]);
            }
        }

        if (!task_description_path.empty()) {
            std::ifstream task_desc_file(task_description_path);
            if (task_desc_file.is_open()) {
                std::stringstream task_desc_buffer;
                task_desc_buffer << task_desc_file.rdbuf();
                const std::string task_desc_json = task_desc_buffer.str();
                const std::string parsed_task = RuntimeHelpers::parseJsonString(task_desc_json, "task", "");
                const std::string parsed_task_name = RuntimeHelpers::parseJsonString(task_desc_json, "task_name", "");
                if (!parsed_task.empty()) {
                    task = RuntimeHelpers::normalizeTaskName(parsed_task, task);
                } else if (!parsed_task_name.empty()) {
                    task = RuntimeHelpers::normalizeTaskName(parsed_task_name, task);
                }
            }
        }

        if (!input_h5_path.empty()) {
            dataset = std::filesystem::path(input_h5_path).stem().string();
        }

        params_path = std::filesystem::path("config") / config_folder;
        config_paths = RuntimeHelpers::getConfigFiles(params_path);
        if (config_paths.empty()) {
            std::cerr << "[ERROR] No JSON config files found at: " << params_path << "\n";
            return 1;
        }

        std::filesystem::path results_csv = std::filesystem::path("results") / (config_folder + ".csv");
        std::filesystem::create_directories(results_csv.parent_path());

        std::ofstream csv_file;
        bool new_results_file = !std::filesystem::exists(results_csv);
        csv_file.open(results_csv, std::ios::app);
        csv_file << std::unitbuf;

        if (!csv_file.is_open()) {
            std::cerr << "[ERROR] Could not open results file: " << results_csv << "\n";
            return 1;
        }
        if (new_results_file) {
            csv_file << "k,itr,nb,nd,heap_factor,mqt,msb,md,"
                     << "Avg_Cluster_Size,Median_Cluster_Size,Avg_Cluster_Similarity,"
                     << "Avg_Blocks_Entered,Avg_Blocks_Skipped,Avg_Docs_Examined,Avg_Docs_Popped,"
                     << "Clustering_Time_s,Indexing_Time_s,"
                     << "Recall@30,Avg_Time_Per_Query_ms,Search_Time_s,Total_Time_s\n";
        }

        // Resolve dataset either from TIRA --input or local fallback layout.
        std::string dataset_path = input_h5_path.empty() ? ("data/" + dataset + ".h5") : input_h5_path;
        std::cout << "[SISAP] Running " << task << " on dataset: " << dataset_path << std::endl;
        std::cout << "[CONFIG] Executing " << config_paths.size() << " config files from `" << params_path.string() << "`" << std::endl;

        std::cout << "\n--- Read data (" << dataset_path << ") ---" << std::endl;
        HDF5SparseLoader loader(dataset_path);

        auto train = loader.load<float>("train");
        auto query = loader.load<float>("otest/queries");

        std::cout << "train: " << train.rows() << " x " << train.cols() << "\n";
        std::cout << "query: " << query.rows() << " x " << query.cols() << "\n";

        for (const auto& config_path : config_paths) {
            std::cout << "\n=====================\n";
            std::cout << "[CONFIG] Running config: " << config_path.string() << "\n";

            std::vector<ExecConfig> exec_configs;
            exec_configs = RuntimeHelpers::loadConfigSetFromFile(config_path.string(), false);

            if (exec_configs.empty()) {
                std::cerr << "[SKIP] No valid configurations could be parsed from " << config_path.string() << "\n";
                continue;
            }
            const ExecConfig base_config = exec_configs.front();

            ExecutionProfiler run_profiler(base_config.mem_debug, base_config.log_debug);
            run_profiler.start("total_run_" + config_path.filename().string());

            std::cout << "[PARAMS] k=" << base_config.num_clusters
                      << " itr=" << base_config.max_iterations
                      << " (base values)\n";

            std::cout << "\n--- Clustering | k = " << base_config.num_clusters
                      << " | iterations = " << base_config.max_iterations << " ---" << std::endl;
            // ExecutionProfiler run_profiler;
            run_profiler.start("clustering");
            ClusteringEngine engine;
            auto result = engine.run(train, base_config, run_profiler);
            run_profiler.stop("clustering");

            std::cout << "\nClustering complete. Sample cluster sizes:\n";
            ClusteringMetrics clustering_metrics = ClusterEvaluator::evaluate(train, result);

            std::cout << "\n--- Summary Vectors for each cluster & Indexing ---" << std::endl;
            // IndexManagerOptimized index_manager;
            IndexManagerSimple index_manager;
            IndexManager& index_manager_ref = index_manager;
            run_profiler.start("computing_summaries");
            std::vector<Eigen::VectorXf> summaries = index_manager_ref.computeSummaryVectors(train, result.assignments, base_config.num_clusters);
            run_profiler.stop("computing_summaries");

            auto gold_standard = loader.loadGoldStandard("otest/knns");
            int num_eval_queries = query.rows();
            double clustering_time_sec = static_cast<double>(run_profiler.getDuration("clustering")) / 1000.0;
            
            std::cout << "\n--- Indexing | nb = " << base_config.max_blocks_per_dimension
                      << " | nd = " << base_config.max_docs_per_block << " ---" << std::endl;
            run_profiler.start("building_index");
            auto inverted_index = index_manager_ref.buildInvertedIndex(train, result.assignments, base_config.num_clusters, base_config, summaries);
            run_profiler.stop("building_index");
            std::cout << "Total Summary Vectors: " << summaries.size() << std::endl;
            std::cout << "Total Concepts indexed: " << inverted_index.size() << std::endl;

            for (const auto& exec_config : exec_configs) {
                std::cout << "\n[RUN starts] k=" << exec_config.num_clusters
                          << " itr=" << exec_config.max_iterations
                          << " nb=" << exec_config.max_blocks_per_dimension
                          << " nd=" << exec_config.max_docs_per_block
                          << " heap_factor=" << exec_config.heap_factor
                          << " mqt=" << exec_config.max_query_terms
                          << " msb=" << exec_config.max_search_blocks
                          << " md=" << exec_config.max_docs_to_visit
                          << "\n";

                try {
                    SearchEngineOptimized search_engine;
                    // SearchEngineSimple search_engine;
                    SearchEngine& search_engine_ref = search_engine;
                    std::vector<std::vector<std::pair<float, int>>> evaluation_results(num_eval_queries);

                    std::cout << "\n--- SEARCH PHASE ---" << std::endl;
                    float average_recall_30 = 0.0f;
                    run_profiler.start("search_phase");
                    
                    int top_n_to_check= 30;
                    std::cout << "\n--- Evaluating Recall@" << top_n_to_check << " ---\n";
                    float total_recall = 0.0f;
                    run_profiler.start("search_phase_top_n_" + std::to_string(top_n_to_check));
                    
                    for (int actual_query_idx = 0; actual_query_idx < num_eval_queries; ++actual_query_idx) {
                        evaluation_results[actual_query_idx] = search_engine_ref.search(train, inverted_index, summaries, query, actual_query_idx, top_n_to_check, exec_config);
                        const auto& hits = evaluation_results[actual_query_idx];
                        auto gold_start = gold_standard[actual_query_idx].begin();
                        auto gold_end = gold_start + std::min<size_t>(top_n_to_check, gold_standard[actual_query_idx].size());

                        int true_positives = 0;
                        for (int k = 0; k < top_n_to_check; ++k) {
                            int predicted_doc = hits[k].second;
                            bool is_true_positive = (std::find(gold_start, gold_end, predicted_doc) != gold_end);
                            if (is_true_positive) {
                                true_positives++;
                            }
                        }
                        float query_recall = (top_n_to_check > 0) ? static_cast<float>(true_positives) / static_cast<float>(top_n_to_check) : 0.0f;
                        total_recall += query_recall;
                        if ((actual_query_idx) % 1000 == 0) {
                            std::cout << "  Processed " << (actual_query_idx) << "/" << num_eval_queries << " queries...\n";
                        }
                    }

                    run_profiler.stop("search_phase_top_n_" + std::to_string(top_n_to_check));
                    average_recall_30 = total_recall / static_cast<float>(num_eval_queries);
                    std::cout << "====================================================\n";
                    // std::cout << "  VAL RESULTS (N = " << num_eval_queries << " queries || MaxDocs = " << exec_config.max_docs_to_visit << ")\n";
                    std::cout << "\n[RUN] k=" << exec_config.num_clusters
                        << " itr=" << exec_config.max_iterations
                        << " nb=" << exec_config.max_blocks_per_dimension
                        << " nd=" << exec_config.max_docs_per_block
                        << " heap_factor=" << exec_config.heap_factor
                        << " mqt=" << exec_config.max_query_terms
                        << " msb=" << exec_config.max_search_blocks
                        << " md=" << exec_config.max_docs_to_visit
                        << "\n";
                    std::cout << "  Average Recall@" << top_n_to_check << " = " << average_recall_30 << "\n";
                    if (average_recall_30 >= 0.90f) {
                        std::cout << "  STATUS: SUCCESS >= 0.90\n";
                    } else {
                        std::cout << "  STATUS: FAIL < 0.90 \n";
                    }
                    std::cout << "====================================================\n";

                    run_profiler.stop("search_phase");

                    double avg_blocks_entered = 0.0;
                    double avg_blocks_skipped = 0.0;
                    double avg_docs_examined = 0.0;
                    double avg_docs_popped = 0.0;
                    search_engine_ref.getAvgDebugStats(avg_blocks_entered, avg_blocks_skipped, avg_docs_examined, avg_docs_popped);
                    search_engine_ref.printAvgDebugStats();

                    double indexing_time_sec = static_cast<double>(run_profiler.getDuration("building_index")) / 1000.0;
                    long long search_time = run_profiler.getDuration("search_phase");
                    double avg_time_per_query_ms = (num_eval_queries > 0) ? static_cast<double>(search_time) / num_eval_queries : 0.0;
                    std::cout << "\n[TIME-QUERY] Average Time per Query: " << avg_time_per_query_ms << " ms\n";
                    double total_time_sec = clustering_time_sec + indexing_time_sec + (static_cast<double>(search_time) / 1000.0);

                    const std::string algo_name = "chnsw";
                    const std::string params_str = RuntimeHelpers::buildParamsString(exec_config);
                    const std::filesystem::path sisap_output_path = RuntimeHelpers::buildSisapResultPath(
                        output_root,
                        task,
                        algo_name,
                        dataset,
                        exec_config,
                        true
                    );
                    const double build_time_seconds = clustering_time_sec + indexing_time_sec;
                    const double query_time_seconds = static_cast<double>(search_time) / 1000.0;
                    RuntimeHelpers::writeSisapResultH5(
                        sisap_output_path,
                        evaluation_results,
                        top_n_to_check,
                        algo_name,
                        task,
                        build_time_seconds,
                        query_time_seconds,
                        params_str
                    );
                    std::cout << "[OK] SISAP HDF5 saved to " << sisap_output_path << "\n";

                        csv_file << exec_config.num_clusters << ","
                            << exec_config.max_iterations << ","
                            << exec_config.max_blocks_per_dimension << ","
                            << exec_config.max_docs_per_block << ","
                            << exec_config.heap_factor << ","
                            << exec_config.max_query_terms << ","
                            << exec_config.max_search_blocks << ","
                            << exec_config.max_docs_to_visit << ","
                            << clustering_metrics.avg_cluster_size << ","
                            << clustering_metrics.median_cluster_size << ","
                            << clustering_metrics.avg_intra_cluster_similarity << ","
                            << avg_blocks_entered << ","
                            << avg_blocks_skipped << ","
                            << avg_docs_examined << ","
                            << avg_docs_popped << ","
                            << clustering_time_sec << ","
                            << indexing_time_sec << ","
                            << average_recall_30 << ","
                            << avg_time_per_query_ms << ","
                            << search_time / 1000.0 << ","
                            << total_time_sec << std::endl;

                        std::cout << "\n[OK] Results saved to " << results_csv << "\n";
                } catch (const std::bad_alloc& e) {
                    std::cerr << "[SKIP] Run skipped due to memory allocation failure: " << e.what() << "\n";
                    continue;
                } catch (const std::runtime_error& e) {
                    std::string msg = e.what();
                    if (msg.find("memory") != std::string::npos || msg.find("Estimated index build memory") != std::string::npos) {
                        std::cerr << "[SKIP] Run skipped due to estimated memory limit: " << e.what() << "\n";
                        continue;
                    }
                    throw;
                }
            }
            run_profiler.stop("total_run_" + config_path.filename().string());
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}