#ifndef BOSTON_DATASET_H
#define BOSTON_DATASET_H


#include "DataLoader.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>


class BostonDataset : public Dataset {
private:
    std::vector<std::vector<float>> features;
    std::vector<float> targets; 
    size_t num_features_ = 0;
    bool is_normalized = false;

    std::vector<float> feature_mean;
    std::vector<float> feature_std;
    float target_mean = 0.0f;
    float target_std = 1.0f;

    // --- Private Helper Methods (load_from_csv, normalization methods) ---
    inline void load_from_csv(const std::string& csv_path) {
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open Boston dataset file: " + csv_path);
        }
        std::string line;
        if (!std::getline(file, line)) {
            throw std::runtime_error("Could not read header line from Boston dataset file.");
        }
        features.clear(); targets.clear();
        while (std::getline(file, line)) {
            std::stringstream ss(line); std::string cell;
            std::vector<float> row_features; float target_value;
            if (!std::getline(ss, cell, ',')) continue; 
            bool row_ok = true;
            for (int i = 0; i < 13; ++i) { 
                if (!std::getline(ss, cell, ',')) { row_ok = false; break; }
                try { row_features.push_back(std::stof(cell)); }
                catch (const std::exception&) { row_ok = false; break; }
            }
            if (!row_ok || !std::getline(ss, cell, ',')) continue;
            try { target_value = std::stof(cell); }
            catch (const std::exception&) { continue; }
            features.push_back(row_features); targets.push_back(target_value);
        }
        if (!features.empty()) num_features_ = features[0].size(); else num_features_ = 0;
        file.close();
    }

    inline void calculate_and_apply_normalization() 
    {
        if (features.empty()) return;
        size_t n_samples = features.size();
        feature_mean.assign(num_features_, 0.0f); feature_std.assign(num_features_, 0.0f);
        target_mean = 0.0f; target_std = 0.0f;
        for (const auto& f : features) 
        { 
            for (size_t j = 0; j < num_features_; ++j) feature_mean[j] += f[j]; 
        }
        target_mean = std::accumulate(targets.begin(), targets.end(), 0.0f);
        for (size_t j = 0; j < num_features_; ++j) feature_mean[j] /= n_samples;

        target_mean /= n_samples;
        for (const auto& f : features) 
        { 
            for (size_t j = 0; j < num_features_; ++j) feature_std[j] += (f[j] - feature_mean[j]) * (f[j] - feature_mean[j]); 
        }
        for (float t : targets) target_std += (t - target_mean) * (t - target_mean);
        for (size_t j = 0; j < num_features_; ++j) 
        { 
            feature_std[j] = std::sqrt(feature_std[j] / n_samples); 
            if (feature_std[j] < 1e-6f) feature_std[j] = 1.0f; 
        }

        target_std = std::sqrt(target_std / n_samples); 
        if (target_std < 1e-6f) target_std = 1.0f;

        normalize_features(feature_mean, feature_std); 
        normalize_targets(target_mean, target_std);
        is_normalized = true;
    }

    inline void normalize_features(const std::vector<float>& mean, const std::vector<float>& stddev) 
    {
         for (auto& sample_features : features) {
            for (size_t j = 0; j < num_features_; ++j) {
                if (stddev[j] != 0.0f) sample_features[j] = (sample_features[j] - mean[j]) / stddev[j];
                else sample_features[j] = 0.0f;
            }
        }
    }

    inline void normalize_targets(float mean, float stddev) 
    {
        if (stddev == 0.0f) 
        { 
            std::fill(targets.begin(), targets.end(), 0.0f); 
            return; 
        }
        for (float& target_val : targets) target_val = (target_val - mean) / stddev;
    }


public:
    // Constructor to load from CSV
    explicit inline BostonDataset(const std::string& csv_path, bool normalize = true) 
    {
        load_from_csv(csv_path);
        if (normalize && !features.empty()) 
        {
            calculate_and_apply_normalization();
        } 
        else 
        {
            is_normalized = false;
            if(num_features_ > 0) 
            { 
                feature_mean.assign(num_features_, 0.0f); 
                feature_std.assign(num_features_, 1.0f); 
            }
            target_mean = 0.0f; target_std = 1.0f;
        }
    }

    // Constructor for pre-split data
    inline BostonDataset(
        std::vector<std::vector<float>> preloaded_features,
        std::vector<float> preloaded_targets,
        bool apply_norm = true,
        const std::vector<float>& train_feature_mean = {},
        const std::vector<float>& train_feature_std = {},
        float train_target_mean = 0.0f,
        float train_target_std = 1.0f
    ) : features(std::move(preloaded_features)), targets(std::move(preloaded_targets))
    {
        if (!features.empty()) num_features_ = features[0].size(); else num_features_ = 0;
        if (features.size() != targets.size()) throw std::runtime_error("Feature/target size mismatch");

        if (apply_norm && !features.empty()) {
            if (train_feature_mean.empty() || train_feature_std.empty()) {
                calculate_and_apply_normalization();
            } else {
                if (train_feature_mean.size() != num_features_ || train_feature_std.size() != num_features_) throw std::runtime_error("Norm stats dim mismatch");
                feature_mean = train_feature_mean; feature_std = train_feature_std;
                target_mean = train_target_mean; target_std = train_target_std;
                normalize_features(feature_mean, feature_std); normalize_targets(target_mean, target_std);
                is_normalized = true;
            }
        } else {
            is_normalized = false;
            feature_mean = train_feature_mean.empty() ? std::vector<float>(num_features_, 0.0f) : train_feature_mean;
            feature_std = train_feature_std.empty() ? std::vector<float>(num_features_, 1.0f) : train_feature_std;
            target_mean = train_target_mean; target_std = train_target_std;
        }
    }


    inline ~BostonDataset() override = default;

    inline size_t size() const override 
    {
        return features.size();
    }

    // Fills out_features and out_targets for the given index
    inline void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const override 
    {
        if (index >= size()) 
        {
            throw std::out_of_range("Index out of range in BostonDataset::get_sample");
        }
        out_features = features[index];
        // Ensure target is always returned as a vector (even if single element)
        out_targets.resize(1);
        out_targets[0] = targets[index];
    }

    inline size_t feature_size() const override 
    {
        return num_features_;
    }

    inline size_t target_size() const override 
    {
        // Boston target is a single value
        return 1;
    }

    // --- Keep existing public methods ---
    inline std::vector<std::vector<float>> get_all_features() const { return features; }
    inline std::vector<float> get_all_targets() const { return targets; }
    inline std::vector<float> get_feature_mean() const { return feature_mean; }
    inline std::vector<float> get_feature_std() const { return feature_std; }
    inline float get_target_mean() const { return target_mean; }
    inline float get_target_std() const { return target_std; }

    // --- Denormalization methods (keep as before) ---
    inline float denormalize_target(float norm_target) const 
    {
        if (!is_normalized) return norm_target;
        return norm_target * target_std + target_mean;
    }

    inline std::vector<float> denormalize_targets(const std::vector<float>& norm_targets) const 
    {
        if (!is_normalized) return norm_targets;
        std::vector<float> denorm_targets = norm_targets;
        std::transform(denorm_targets.begin(), denorm_targets.end(), denorm_targets.begin(),
                       [this](float t) { return denormalize_target(t); });
        return denorm_targets;
    }
    // Denormalize features (if needed)
    inline std::vector<float> denormalize_features(const std::vector<float>& norm_features) const 
    {
        if (!is_normalized) return norm_features;
        if (norm_features.size() != num_features_) throw std::runtime_error("Feature size mismatch in denormalize_features");
        std::vector<float> denorm_features = norm_features;
        for (size_t j = 0; j < num_features_; ++j) 
        {
            denorm_features[j] = norm_features[j] * feature_std[j] + feature_mean[j];
        }
        return denorm_features;
    }
};

#endif // BOSTON_DATASET_H