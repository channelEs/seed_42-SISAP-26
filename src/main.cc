#include <iostream>
#include <fstream>
#include <sstream>
#include <new>
#include "utils/hdf5_sparse_loader.h"
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
#include <limits>
#include <cstdint>
#include <Eigen/Sparse>
#include <hdf5.h>

bool DEV = false;

static const std::string STATIC_DATASET = "nq";
static const std::string STATIC_TASK = "task3";
static const std::string STATIC_RESULTS_FILE = "static_prod.csv";

static ExecConfig getStaticExecConfig() {
    ExecConfig cfg;
    cfg.num_clusters = 2000;
    cfg.max_iterations = 3;
    cfg.max_blocks_per_dimension = 250;
    cfg.max_docs_per_block = 150;
    cfg.max_docs_to_visit = 60000;
    cfg.heap_factor = 0.15f;
    cfg.log_debug = false;
    cfg.mem_debug = false;
    return cfg;
}

static std::string sanitizeFilenameToken(const std::string& value) {
    std::string out = value;
    for (char& ch : out) {
        const bool is_alnum = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
        if (!is_alnum && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return out;
}

static std::string toCompactFloatToken(float value, int precision = 3) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(precision);
    oss << value;
    std::string token = oss.str();
    while (!token.empty() && token.back() == '0') {
        token.pop_back();
    }
    if (!token.empty() && token.back() == '.') {
        token.pop_back();
    }
    std::replace(token.begin(), token.end(), '.', 'p');
    std::replace(token.begin(), token.end(), '-', 'm');
    return token.empty() ? "0" : token;
}

static std::string buildParamsString(const ExecConfig& cfg) {
    std::ostringstream params;
    params << "k=" << cfg.num_clusters
           << ",itr=" << cfg.max_iterations
           << ",nb=" << cfg.max_blocks_per_dimension
           << ",nd=" << cfg.max_docs_per_block
           << ",md=" << cfg.max_docs_to_visit
           << ",heap_factor=" << cfg.heap_factor;
    return params.str();
}

static std::filesystem::path buildSisapResultPath(
    const std::string& task,
    const std::string& algo,
    const std::string& dataset,
    const ExecConfig& cfg
) {
    const std::string filename = sanitizeFilenameToken(algo)
        + "_"
        + sanitizeFilenameToken(dataset)
        + "_k" + std::to_string(cfg.num_clusters)
        + "_itr" + std::to_string(cfg.max_iterations)
        + "_nb" + std::to_string(cfg.max_blocks_per_dimension)
        + "_nd" + std::to_string(cfg.max_docs_per_block)
        + "_md" + std::to_string(cfg.max_docs_to_visit)
        + "_hf" + toCompactFloatToken(cfg.heap_factor)
        + ".h5";

    return std::filesystem::path("results") / task / filename;
}

static void writeStringRootAttribute(hid_t file, const char* key, const std::string& value) {
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, value.size() + 1);
    H5Tset_strpad(str_type, H5T_STR_NULLTERM);

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(file, key, str_type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, str_type, value.c_str());

    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(str_type);
}

template <typename T>
static void writeNumericRootAttribute(hid_t file, const char* key, hid_t type, const T& value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(file, key, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

static void writeSisapResultH5(
    const std::filesystem::path& output_path,
    const std::vector<std::vector<std::pair<float, int>>>& evaluation_results,
    int top_k,
    const std::string& algo,
    const std::string& task,
    double build_time_seconds,
    double query_time_seconds,
    const std::string& params
) {
    const int n_queries = static_cast<int>(evaluation_results.size());
    if (n_queries <= 0 || top_k <= 0) {
        throw std::runtime_error("Invalid shape for SISAP result export.");
    }

    std::filesystem::create_directories(output_path.parent_path());

    const size_t total = static_cast<size_t>(n_queries) * static_cast<size_t>(top_k);
    std::vector<int32_t> knns_flat(total, 0);
    std::vector<float> dists_flat(total, std::numeric_limits<float>::infinity());

    for (int q = 0; q < n_queries; ++q) {
        const auto& hits = evaluation_results[q];
        const int filled = std::min<int>(top_k, static_cast<int>(hits.size()));
        for (int i = 0; i < filled; ++i) {
            const size_t pos = static_cast<size_t>(q) * static_cast<size_t>(top_k) + static_cast<size_t>(i);
            const int doc_id_zero_based = hits[i].second;
            knns_flat[pos] = (doc_id_zero_based >= 0) ? static_cast<int32_t>(doc_id_zero_based + 1) : 0;
            dists_flat[pos] = hits[i].first;
        }
    }

    hid_t file = H5Fcreate(output_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("Failed to create HDF5 result file: " + output_path.string());
    }

    hsize_t dims[2] = {static_cast<hsize_t>(n_queries), static_cast<hsize_t>(top_k)};
    hid_t space = H5Screate_simple(2, dims, nullptr);

    hid_t knns_ds = H5Dcreate2(file, "knns", H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dists_ds = H5Dcreate2(file, "dists", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(knns_ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, knns_flat.data());
    H5Dwrite(dists_ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dists_flat.data());

    writeStringRootAttribute(file, "algo", algo);
    writeStringRootAttribute(file, "task", task);
    writeStringRootAttribute(file, "params", params);
    writeNumericRootAttribute<double>(file, "buildtime", H5T_NATIVE_DOUBLE, build_time_seconds);
    writeNumericRootAttribute<double>(file, "querytime", H5T_NATIVE_DOUBLE, query_time_seconds);

    H5Dclose(knns_ds);
    H5Dclose(dists_ds);
    H5Sclose(space);
    H5Fclose(file);
}

static int parseJsonInt(const std::string& json, const std::string& key, int defaultValue) {
    if (json.empty()) {
        return defaultValue;
    }

    std::string quotedKey = "\"" + key + "\"";
    auto pos = json.find(quotedKey);
    if (pos == std::string::npos) {
        pos = json.find(key);
        if (pos == std::string::npos) {
            return defaultValue;
        }
    }

    auto colon = json.find(':', pos);
    if (colon == std::string::npos) {
        return defaultValue;
    }

    auto start = json.find_first_of("-0123456789", colon + 1);
    if (start == std::string::npos) {
        return defaultValue;
    }

    auto end = json.find_first_not_of("0123456789", start + 1);
    std::string token = json.substr(start, end == std::string::npos ? json.size() - start : end - start);
    try {
        return std::stoi(token);
    } catch (...) {
        return defaultValue;
    }
}

static float parseJsonFloat(const std::string& json, const std::string& key, float defaultValue) {
    if (json.empty()) {
        return defaultValue;
    }

    std::string quotedKey = "\"" + key + "\"";
    auto pos = json.find(quotedKey);
    if (pos == std::string::npos) {
        pos = json.find(key);
        if (pos == std::string::npos) {
            return defaultValue;
        }
    }

    auto colon = json.find(':', pos);
    if (colon == std::string::npos) {
        return defaultValue;
    }

    auto start = json.find_first_of("-0123456789.", colon + 1);
    if (start == std::string::npos) {
        return defaultValue;
    }

    auto end = json.find_first_not_of("0123456789.", start + 1);
    std::string token = json.substr(start, end == std::string::npos ? json.size() - start : end - start);
    try {
        return std::stof(token);
    } catch (...) {
        return defaultValue;
    }
}

static std::vector<float> parseJsonFloatArray(const std::string& json, const std::string& key) {
    std::vector<float> values;
    if (json.empty()) {
        return values;
    }

    std::string quotedKey = "\"" + key + "\"";
    auto pos = json.find(quotedKey);
    if (pos == std::string::npos) {
        pos = json.find(key);
        if (pos == std::string::npos) {
            return values;
        }
    }

    auto colon = json.find(':', pos);
    if (colon == std::string::npos) {
        return values;
    }

    auto start = json.find_first_of("[", colon + 1);
    if (start == std::string::npos) {
        return values;
    }

    auto end = json.find(']', start + 1);
    if (end == std::string::npos) {
        return values;
    }

    std::string array_content = json.substr(start + 1, end - start - 1);
    std::stringstream ss(array_content);
    float value;
    char separator;
    while (ss >> value) {
        values.push_back(value);
        ss >> separator;
    }

    return values;
}

static std::vector<int> parseJsonIntArray(const std::string& json, const std::string& key) {
    std::vector<int> values;
    if (json.empty()) {
        return values;
    }

    std::string quotedKey = "\"" + key + "\"";
    auto pos = json.find(quotedKey);
    if (pos == std::string::npos) {
        pos = json.find(key);
        if (pos == std::string::npos) {
            return values;
        }
    }

    auto colon = json.find(':', pos);
    if (colon == std::string::npos) {
        return values;
    }

    auto start = json.find('[', colon + 1);
    if (start == std::string::npos) {
        return values;
    }

    auto end = json.find(']', start + 1);
    if (end == std::string::npos) {
        return values;
    }

    std::string array_content = json.substr(start + 1, end - start - 1);
    std::stringstream ss(array_content);
    int value;
    char separator;
    while (ss >> value) {
        values.push_back(value);
        ss >> separator;
    }

    return values;
}

static std::vector<ExecConfig> loadConfigSetFromFile(const std::string& filepath) {
    std::vector<ExecConfig> configs;
    ExecConfig baseConfig;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[WARNING] Could not open config file: " << filepath << ". Using defaults.\n";
        return configs;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_content = buffer.str();
    file.close();

    baseConfig.num_clusters = parseJsonInt(json_content, "k", 200);
    baseConfig.max_iterations = parseJsonInt(json_content, "itr", 3);
    if (DEV) {
        baseConfig.log_debug = true;
    } else {
        baseConfig.log_debug = false;
    }
    baseConfig.mem_debug = false;

    auto nbs = parseJsonIntArray(json_content, "nb");
    if (nbs.empty()) {
        nbs.push_back(parseJsonInt(json_content, "nb", 0));
    }
    
    auto nds = parseJsonIntArray(json_content, "nd");
    if (nds.empty()) {
        nds.push_back(parseJsonInt(json_content, "nd", 0));
    }
    
    auto mds = parseJsonIntArray(json_content, "md");
    if (mds.empty()) {
        mds.push_back(parseJsonInt(json_content, "md", 0));
    }

    auto heap_factors = parseJsonFloatArray(json_content, "heap_factor");
    if (heap_factors.empty()) {
        heap_factors.push_back(parseJsonFloat(json_content, "heap_factor", 0.15f));
    }

    for (int nb : nbs) {
        for (int nd : nds) {
            for (int md : mds) {
                for (float heap_factor : heap_factors) {
                    ExecConfig cfg = baseConfig;
                    cfg.max_blocks_per_dimension = nb;
                    cfg.max_docs_per_block = nd;
                    cfg.max_docs_to_visit = md;
                    cfg.heap_factor = heap_factor;
                    configs.push_back(cfg);
                }
            }
        }
    }

    return configs;
}

static std::vector<std::filesystem::path> getConfigFiles(const std::filesystem::path& params_path) {
    std::vector<std::filesystem::path> config_files;

    if (std::filesystem::is_directory(params_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(params_path)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().extension() == ".json") {
                config_files.push_back(entry.path());
            }
        }
        std::sort(config_files.begin(), config_files.end());
    } else {
        config_files.push_back(params_path);
    }

    return config_files;
}

int main(int argc, char* argv[]) {
    try {
        std::set_new_handler([]() {
            std::cerr << "[ERROR] Memory allocation failure: std::new_handler triggered.\n";
        });

        // Default fallbacks if flags aren't passed
        std::string dataset = STATIC_DATASET;
        std::string task = STATIC_TASK;
        std::string config_folder = "clusters";

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
            }
        }
        if (DEV) {
            params_path = std::filesystem::path("config") / config_folder;
            config_paths = getConfigFiles(params_path);
            if (config_paths.empty()) {
                std::cerr << "[ERROR] No JSON config files found at: " << params_path << "\n";
                return 1;
            }
        } else {
            config_paths.push_back("STATIC_PROD_CONFIG");
        }

        std::filesystem::path results_csv = DEV
            ? (std::filesystem::path("results") / (config_folder + ".csv"))
            : (std::filesystem::path("results") / STATIC_RESULTS_FILE);

        std::ofstream csv_file;
        if (DEV) {
            bool new_results_file = !std::filesystem::exists(results_csv);
            csv_file.open(results_csv, std::ios::app);
            csv_file << std::unitbuf;
            
            if (!csv_file.is_open()) {
                std::cerr << "[ERROR] Could not open results file: " << results_csv << "\n";
                return 1;
            }
            if (new_results_file) {
                csv_file << "k,itr,nb,nd,md,heap_factor,"
                         << "Avg_Cluster_Size,Median_Cluster_Size,Avg_Cluster_Similarity,"
                         << "Avg_Blocks_Entered,Avg_Blocks_Skipped,Avg_Docs_Examined,Avg_Docs_Popped,"
                         << "Clustering_Time_s,Indexing_Time_s,"
                         << "Recall@30,Avg_Time_Per_Query_ms,Search_Time_s,Total_Time_s\n";
            }
        }


        // Dynamically resolve dataset file pathway using the mount layout
        std::string dataset_path = "data/" + dataset + ".h5";
        std::cout << "[SISAP] Running " << task << " on dataset: " << dataset_path << std::endl;
        if (DEV) {
            std::cout << "[CONFIG] Executing " << config_paths.size() << " config files from `" << params_path.string() << "`" << std::endl;
        } else {
            std::cout << "[CONFIG] Production exec" << std::endl;
        }

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
            if (DEV) {
                exec_configs = loadConfigSetFromFile(config_path.string());
            } else {
                exec_configs.push_back(getStaticExecConfig());
            }

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

            for (const auto& exec_config : exec_configs) {
                std::cout << "\n[RUN starts] k=" << exec_config.num_clusters
                          << " itr=" << exec_config.max_iterations 
                          << " nb=" << exec_config.max_blocks_per_dimension
                          << " nd=" << exec_config.max_docs_per_block
                          << " md=" << exec_config.max_docs_to_visit 
                          << " heap_factor=" << exec_config.heap_factor
                          << "\n";

                try {
                    // ExecutionProfiler run_profiler;
                    std::cout << "\n--- Indexing | nb = " << exec_config.max_blocks_per_dimension
                              << " | nd = " << exec_config.max_docs_per_block << " ---" << std::endl;
                    run_profiler.start("building_index");
                    auto inverted_index = index_manager_ref.buildInvertedIndex(train, result.assignments, exec_config.num_clusters, exec_config, summaries);
                    run_profiler.stop("building_index");

                    std::cout << "Total Summary Vectors: " << summaries.size() << std::endl;
                    std::cout << "Total Concepts indexed: " << inverted_index.size() << std::endl;

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
                        << " md=" << exec_config.max_docs_to_visit 
                        << " heap_factor=" << exec_config.heap_factor
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
                    const std::string params_str = buildParamsString(exec_config);
                    const std::filesystem::path sisap_output_path = buildSisapResultPath(task, algo_name, dataset, exec_config);
                    const double build_time_seconds = clustering_time_sec + indexing_time_sec;
                    const double query_time_seconds = static_cast<double>(search_time) / 1000.0;
                    writeSisapResultH5(
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

                    if (DEV) {
                        csv_file << exec_config.num_clusters << ","
                                << exec_config.max_iterations << ","
                                << exec_config.max_blocks_per_dimension << ","
                                << exec_config.max_docs_per_block << ","
                                << exec_config.max_docs_to_visit << ","
                                << exec_config.heap_factor << ","
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
                    }
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