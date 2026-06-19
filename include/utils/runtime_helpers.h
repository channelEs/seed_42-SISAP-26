#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "utils/types.h"

namespace RuntimeHelpers {

ExecConfig getStaticExecConfig();
std::string buildParamsString(const ExecConfig& cfg);
std::filesystem::path buildSisapResultPath(
    const std::filesystem::path& output_root,
    const std::string& task,
    const std::string& algo,
    const std::string& dataset,
    const ExecConfig& cfg,
    bool include_task_subdir
);
std::string parseJsonString(const std::string& json, const std::string& key, const std::string& default_value);
std::string normalizeTaskName(const std::string& raw_task, const std::string& fallback_task);
void writeSisapResultH5(
    const std::filesystem::path& output_path,
    const std::vector<std::vector<std::pair<float, int>>>& evaluation_results,
    int top_k,
    const std::string& algo,
    const std::string& task,
    double build_time_seconds,
    double query_time_seconds,
    const std::string& params
);
std::vector<ExecConfig> loadConfigSetFromFile(const std::string& filepath, bool log_debug);
std::vector<std::filesystem::path> getConfigFiles(const std::filesystem::path& params_path);

}  // namespace RuntimeHelpers
