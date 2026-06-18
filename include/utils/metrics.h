#pragma once
#include <iostream>
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <cctype>
#include "utils/types.h"

static uint64_t parseUnsignedLongLong(const std::string& text) {
    try {
        size_t pos = 0;
        return std::stoull(text, &pos, 10);
    } catch (...) {
        return 0;
    }
}

static uint64_t getCurrentProcessRSSBytes() {
    std::ifstream file("/proc/self/status");
    if (!file.is_open()) {
        return 0;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            auto colon = line.find(':');
            if (colon == std::string::npos) {
                return 0;
            }
            std::string value = line.substr(colon + 1);
            size_t pos = 0;
            while (pos < value.size() && std::isspace(static_cast<unsigned char>(value[pos]))) {
                ++pos;
            }
            size_t end = pos;
            while (end < value.size() && std::isdigit(static_cast<unsigned char>(value[end]))) {
                ++end;
            }
            uint64_t kb = parseUnsignedLongLong(value.substr(pos, end - pos));
            return kb * 1024ULL;
        }
    }

    return 0;
}

static uint64_t readUnsignedFromFile(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }

    std::string line;
    if (!std::getline(file, line)) {
        return 0;
    }

    if (line == "max") {
        return 0;
    }

    return parseUnsignedLongLong(line);
}

static uint64_t getCgroupMemoryLimitBytes() {
    uint64_t limit = readUnsignedFromFile("/sys/fs/cgroup/memory.max");
    if (limit > 0) {
        return limit;
    }
    limit = readUnsignedFromFile("/sys/fs/cgroup/memory.limit_in_bytes");
    return limit;
}

static uint64_t getPhysicalMemoryBytes() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
}

static uint64_t getEffectiveMemoryLimitBytes() {
    uint64_t cgroup_limit = getCgroupMemoryLimitBytes();
    if (cgroup_limit > 0 && cgroup_limit < std::numeric_limits<uint64_t>::max()) {
        return cgroup_limit;
    }
    return getPhysicalMemoryBytes();
}

static double bytesToGB(uint64_t bytes) {
    const double gb = 1024.0 * 1024.0 * 1024.0;
    return static_cast<double>(bytes) / gb;
}

class ExecutionProfiler {
    bool mem_debug = false;

public:
    ExecutionProfiler() = default;
    ExecutionProfiler(bool mem_debug) : mem_debug(mem_debug) {};
    
    void start(const std::string& tag) {
        start_times[tag] = std::chrono::high_resolution_clock::now();
        start_memory[tag] = getCurrentProcessRSSBytes();
        uint64_t limit = getEffectiveMemoryLimitBytes();
        std::cout << "[PROFILER_START] " << tag
                  << " memory_start=" << start_memory[tag] << " bytes (" << bytesToGB(start_memory[tag]) << " GB)"
                  << " limit=" << limit << " bytes (" << bytesToGB(limit) << " GB)" << std::endl;
    }

    void stop(const std::string& tag) {
        auto end = std::chrono::high_resolution_clock::now();
        auto start = start_times[tag];
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        durations[tag] = duration;
        uint64_t end_mem = getCurrentProcessRSSBytes();
        uint64_t start_mem = 0;
        auto mem_it = start_memory.find(tag);
        if (mem_it != start_memory.end()) {
            start_mem = mem_it->second;
        }
        int64_t delta = static_cast<int64_t>(end_mem) - static_cast<int64_t>(start_mem);
        uint64_t limit = getEffectiveMemoryLimitBytes();
        std::cout << "[PROFILER_STOP] " << tag << ": " << duration << " ms (" << duration / 60000.0 << " minutes)" << std::endl;
        
        if (mem_debug) {
            std::cout << "[PROFILER_MEM] " << tag
                << " start=" << start_mem << " bytes (" << bytesToGB(start_mem) << " GB)"
                << " stop=" << end_mem << " bytes (" << bytesToGB(end_mem) << " GB)"
                << " delta=" << delta << " bytes (" << bytesToGB((delta >= 0) ? static_cast<uint64_t>(delta) : static_cast<uint64_t>(-delta)) << " GB)"
                << " limit=" << limit << " bytes (" << bytesToGB(limit) << " GB)"
                << std::endl;
        }
        memory_delta[tag] = delta;
    }

    long long getDuration(const std::string& tag) const {
        auto it = durations.find(tag);
        return (it != durations.end()) ? it->second : 0;
    }

    int64_t getMemoryDelta(const std::string& tag) const {
        auto it = memory_delta.find(tag);
        return (it != memory_delta.end()) ? it->second : 0;
    }

private:
    std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> start_times;
    std::map<std::string, long long> durations;
    std::map<std::string, int64_t> memory_delta;
    std::map<std::string, uint64_t> start_memory;
};

struct ClusteringMetrics {
    double avg_cluster_size;
    int median_cluster_size;
    double avg_intra_cluster_similarity;
};

class ClusterEvaluator {
public:
    static ClusteringMetrics evaluate(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& data,
        const ClusterResult& result
    ) 
    {
        int total_docs = data.rows();
        int n_clusters = result.centroids.size();

        double total_sim = 0;
        for (int i = 0; i < total_docs; ++i) {
            int c = result.assignments[i];
            // Dot product between doc and its assigned centroid
            total_sim += data.row(i).dot(result.centroids[c]);
        }
        double avg_intra_cluster_similarity = total_sim / total_docs;

        std::vector<int> counts(n_clusters, 0);
        for (int a : result.assignments) counts[a]++;
        
        // median cluster size
        std::vector<int> sorted_counts = counts;
        std::sort(sorted_counts.begin(), sorted_counts.end());
        int median_cluster_size = sorted_counts[n_clusters / 2];

        int largest_cluster = *std::max_element(counts.begin(), counts.end());
        int smallest_cluster = *std::min_element(counts.begin(), counts.end());
        int empty_clusters = std::count(counts.begin(), counts.end(), 0);

        double avg_cluster_size = static_cast<double>(total_docs) / n_clusters;

        double size_deviation = 0.0;
        for (int size : counts) {
            size_deviation += std::pow(size - avg_cluster_size, 2);
        }
        std::cout << "\n--- Evaluate clustering (relating docs information) ---" << std::endl;
        size_deviation = std::sqrt(size_deviation / n_clusters);
        std::cout << "Objective Average Cluster Size: " << avg_cluster_size << "\n";
        std::cout << "Median Cluster Size: " << median_cluster_size << "\n";
        std::cout << "Largest Cluster Size: " << largest_cluster << "\n";
        std::cout << "Smallest Cluster Size: " << smallest_cluster << "\n";
        std::cout << "Empty Clusters: " << empty_clusters << "\n";
        std::cout << "Cluster Size Deviation: " << size_deviation << "\n";  
        std::cout << "Avg Similarity (Docs and its cluster assignments with dot product): " << avg_intra_cluster_similarity << std::endl;

        return ClusteringMetrics{avg_cluster_size, median_cluster_size, avg_intra_cluster_similarity};
    }
};