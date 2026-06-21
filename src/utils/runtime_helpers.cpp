#include "utils/runtime_helpers.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <hdf5.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace RuntimeHelpers {
namespace {

std::string sanitizeFilenameToken(const std::string& value) {
    std::string out = value;
    for (char& ch : out) {
        const bool is_alnum = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
        if (!is_alnum && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return out;
}

std::string toCompactFloatToken(float value, int precision = 3) {
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

int parseJsonInt(const std::string& json, const std::string& key, int default_value) {
    if (json.empty()) {
        return default_value;
    }

    std::string quoted_key = "\"" + key + "\"";
    auto pos = json.find(quoted_key);
    if (pos == std::string::npos) {
        pos = json.find(key);
        if (pos == std::string::npos) {
            return default_value;
        }
    }

    auto colon = json.find(':', pos);
    if (colon == std::string::npos) {
        return default_value;
    }

    auto start = json.find_first_of("-0123456789", colon + 1);
    if (start == std::string::npos) {
        return default_value;
    }

    auto end = json.find_first_not_of("0123456789", start + 1);
    std::string token = json.substr(start, end == std::string::npos ? json.size() - start : end - start);
    try {
        return std::stoi(token);
    } catch (...) {
        return default_value;
    }
}

float parseJsonFloat(const std::string& json, const std::string& key, float default_value) {
    if (json.empty()) {
        return default_value;
    }

    std::string quoted_key = "\"" + key + "\"";
    auto pos = json.find(quoted_key);
    if (pos == std::string::npos) {
        pos = json.find(key);
        if (pos == std::string::npos) {
            return default_value;
        }
    }

    auto colon = json.find(':', pos);
    if (colon == std::string::npos) {
        return default_value;
    }

    auto start = json.find_first_of("-0123456789.", colon + 1);
    if (start == std::string::npos) {
        return default_value;
    }

    auto end = json.find_first_not_of("0123456789.", start + 1);
    std::string token = json.substr(start, end == std::string::npos ? json.size() - start : end - start);
    try {
        return std::stof(token);
    } catch (...) {
        return default_value;
    }
}

std::vector<int> parseJsonIntArray(const std::string& json, const std::string& key) {
    std::vector<int> values;
    if (json.empty()) {
        return values;
    }

    std::string quoted_key = "\"" + key + "\"";
    auto pos = json.find(quoted_key);
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

std::vector<float> parseJsonFloatArray(const std::string& json, const std::string& key) {
    std::vector<float> values;
    if (json.empty()) {
        return values;
    }

    std::string quoted_key = "\"" + key + "\"";
    auto pos = json.find(quoted_key);
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
    float value;
    char separator;
    while (ss >> value) {
        values.push_back(value);
        ss >> separator;
    }

    return values;
}

void writeStringRootAttribute(hid_t file, const char* key, const std::string& value) {
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
void writeNumericRootAttribute(hid_t file, const char* key, hid_t type, const T& value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(file, key, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

}  // namespace

ExecConfig getStaticExecConfig() {
    ExecConfig cfg;
    cfg.num_clusters = 2000;
    cfg.max_iterations = 3;
    cfg.max_blocks_per_dimension = 1000;
    cfg.max_docs_per_block = 500;
    cfg.heap_factor = 0.15f;
    cfg.max_query_terms = 0;
    cfg.max_search_blocks = 0;
    cfg.max_docs_to_visit = 60000;
    cfg.log_debug = false;
    cfg.mem_debug = false;
    return cfg;
}

std::string buildParamsString(const ExecConfig& cfg) {
    std::ostringstream params;
    params << "k=" << cfg.num_clusters
           << ",itr=" << cfg.max_iterations
           << ",nb=" << cfg.max_blocks_per_dimension
           << ",nd=" << cfg.max_docs_per_block
           << ",hf=" << cfg.heap_factor
           << ",mqt=" << cfg.max_query_terms
           << ",msb=" << cfg.max_search_blocks
           << ",md=" << cfg.max_docs_to_visit;
    return params.str();
}

std::filesystem::path buildSisapResultPath(
    const std::filesystem::path& output_root,
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
        + "_hf" + toCompactFloatToken(cfg.heap_factor)
        + "_mqt" + std::to_string(cfg.max_query_terms)
        + "_msb" + std::to_string(cfg.max_search_blocks)
        + "_md" + std::to_string(cfg.max_docs_to_visit)
        + ".h5";

    return output_root / filename;
}

std::string parseJsonString(const std::string& json, const std::string& key, const std::string& default_value) {
    if (json.empty()) {
        return default_value;
    }

    std::string quoted_key = "\"" + key + "\"";
    auto pos = json.find(quoted_key);
    if (pos == std::string::npos) {
        pos = json.find(key);
        if (pos == std::string::npos) {
            return default_value;
        }
    }

    auto colon = json.find(':', pos);
    if (colon == std::string::npos) {
        return default_value;
    }

    auto first_quote = json.find('"', colon + 1);
    if (first_quote == std::string::npos) {
        return default_value;
    }

    auto second_quote = json.find('"', first_quote + 1);
    if (second_quote == std::string::npos) {
        return default_value;
    }

    return json.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::string normalizeTaskName(const std::string& raw_task, const std::string& fallback_task) {
    if (raw_task.empty()) {
        return fallback_task;
    }

    std::string compact;
    compact.reserve(raw_task.size());
    for (char ch : raw_task) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            compact.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }

    if (compact == "task1") return "task1";
    if (compact == "task2") return "task2";
    if (compact == "task3") return "task3";
    return fallback_task;
}

void writeSisapResultH5(
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

std::vector<ExecConfig> loadConfigSetFromFile(const std::string& filepath, bool log_debug) {
    std::vector<ExecConfig> configs;
    ExecConfig base_config = getStaticExecConfig();

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[WARNING] Could not open config file: " << filepath << ". Using defaults.\n";
        return configs;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_content = buffer.str();
    file.close();

    base_config.num_clusters = parseJsonInt(json_content, "k", base_config.num_clusters);
    base_config.max_iterations = parseJsonInt(json_content, "itr", base_config.max_iterations);
    base_config.max_blocks_per_dimension = parseJsonInt(json_content, "nb", base_config.max_blocks_per_dimension);
    base_config.max_docs_per_block = parseJsonInt(json_content, "nd", base_config.max_docs_per_block);
    base_config.log_debug = log_debug;
    base_config.mem_debug = false;

    auto mds = parseJsonIntArray(json_content, "md");
    if (mds.empty()) {
        mds.push_back(parseJsonInt(json_content, "md", base_config.max_docs_to_visit));
    }

    auto heap_factors = parseJsonFloatArray(json_content, "heap_factor");
    if (heap_factors.empty()) {
        heap_factors.push_back(parseJsonFloat(json_content, "heap_factor", base_config.heap_factor));
    }

    auto mqts = parseJsonIntArray(json_content, "mqt");
    if (mqts.empty()) {
        mqts.push_back(parseJsonInt(json_content, "mqt", base_config.max_query_terms));
    }

    auto msbs = parseJsonIntArray(json_content, "msb");
    if (msbs.empty()) {
        msbs.push_back(parseJsonInt(json_content, "msb", base_config.max_search_blocks));
    }

    for (int md : mds) {
        for (float heap_factor : heap_factors) {
            for (int mqt : mqts) {
                for (int msb : msbs) {
                    ExecConfig cfg = base_config;
                    cfg.max_docs_to_visit = md;
                    cfg.heap_factor = heap_factor;
                    cfg.max_query_terms = mqt;
                    cfg.max_search_blocks = msb;
                    configs.push_back(cfg);
                }
            }
        }
    }

    return configs;
}

std::vector<std::filesystem::path> getConfigFiles(const std::filesystem::path& params_path) {
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

}  // namespace RuntimeHelpers
