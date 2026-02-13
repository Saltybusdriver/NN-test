#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <random>
#include <algorithm>
#include <memory>
#include <atomic>
#include <numeric> 
#include <cstring> 
#include "utils.h"
#include <cuda_runtime.h>
#include "layer_proxy.h" 
#include <sstream> 
#include <system_error> 

class Dataset;


struct Batch {
    std::vector<float> inputs_flattened;
    std::vector<float> targets_flattened;
    size_t batch_size = 0; 

    size_t max_batch_size_ = 0;
    size_t feature_size_hint_ = 0;
    size_t target_size_hint_ = 0;

    float* d_inputs = nullptr;  
    float* d_targets = nullptr; 
    bool is_data_on_device = false;

private:
    float* _allocated_device_inputs = nullptr;
    float* _allocated_device_targets = nullptr;

    size_t _allocated_input_bytes = 0;
    size_t _allocated_target_bytes = 0;

public:
    Batch(size_t actual_size, size_t max_size, size_t feat_hint, size_t tgt_hint)
        : batch_size(actual_size),
          max_batch_size_(max_size),
          feature_size_hint_(feat_hint),
          target_size_hint_(tgt_hint),
          d_inputs(nullptr),
          d_targets(nullptr),
          is_data_on_device(false),
          _allocated_device_inputs(nullptr),
          _allocated_device_targets(nullptr),
          _allocated_input_bytes(0),
          _allocated_target_bytes(0)
    {}
    Batch() = default;

    ~Batch() {
        if (_allocated_device_inputs) 
        {
            safeCudaFree(&_allocated_device_inputs, "Batch internal d_inputs");
        }
        if (_allocated_device_targets) 
        {
            safeCudaFree(&_allocated_device_targets, "Batch internal d_targets");
        }
        d_inputs = nullptr;
        d_targets = nullptr;
        _allocated_device_inputs = nullptr;
        _allocated_device_targets = nullptr;
        is_data_on_device = false;
    }

    void to_device() {
        bool use_cuda = GetUseCuda();

        // --- Handle Empty Batch ---
        if (batch_size == 0 || inputs_flattened.empty()) 
        {

            if (is_data_on_device) 
            {
                 if (_allocated_device_inputs) safeCudaFree(&_allocated_device_inputs, "Batch internal d_inputs cleanup (empty)");
                 if (_allocated_device_targets) safeCudaFree(&_allocated_device_targets, "Batch internal d_targets cleanup (empty)");
                 _allocated_input_bytes = 0;
                 _allocated_target_bytes = 0;
            }
            d_inputs = nullptr;
            d_targets = nullptr;
            is_data_on_device = false;
            return;
        }

        if (use_cuda) 
        {
            if (max_batch_size_ == 0 || feature_size_hint_ == 0) 
            {
                 fprintf(stderr, "[Batch %p] to_device(): ERROR - Cannot allocate GPU memory with max_batch_size=%zu or feature_size_hint=%zu\n",
                        (void*)this, max_batch_size_, feature_size_hint_); fflush(stderr);
                 // Free existing if any
                 if (_allocated_device_inputs) safeCudaFree(&_allocated_device_inputs, "Batch invalid hint cleanup");
                 if (_allocated_device_targets) safeCudaFree(&_allocated_device_targets, "Batch invalid hint cleanup");
                 d_inputs = nullptr; d_targets = nullptr; is_data_on_device = false;
                 _allocated_input_bytes = 0; _allocated_target_bytes = 0;
                 return;
            }

            size_t required_input_bytes = max_batch_size_ * feature_size_hint_ * sizeof(float);
            size_t required_target_bytes = (target_size_hint_ > 0) ? max_batch_size_ * target_size_hint_ * sizeof(float) : 0;

            // --- Check if Reallocation is Needed ---
            bool needs_realloc = false;
            if (!_allocated_device_inputs || _allocated_input_bytes < required_input_bytes) 
            {
                needs_realloc = true;
                if (_allocated_device_inputs) 
                {
                    safeCudaFree(&_allocated_device_inputs, "Batch internal d_inputs realloc");
                    _allocated_input_bytes = 0;
                }
            }
            if (required_target_bytes > 0) 
            {
                 if (!_allocated_device_targets || _allocated_target_bytes < required_target_bytes) 
                 {
                     needs_realloc = true;
                     if (_allocated_device_targets) 
                     {
                         safeCudaFree(&_allocated_device_targets, "Batch internal d_targets realloc");
                         _allocated_target_bytes = 0;
                     }
                 }
            } 
            else 
            {
                 if (_allocated_device_targets) 
                 {
                     safeCudaFree(&_allocated_device_targets, "Batch internal d_targets cleanup (no longer needed)");
                     _allocated_target_bytes = 0;
                 }
            }

            if (needs_realloc) 
            {
                cudaError_t err_in = cudaMalloc(&_allocated_device_inputs, required_input_bytes);
                cudaError_t err_tgt = cudaSuccess;
                if (required_target_bytes > 0) 
                {
                    err_tgt = cudaMalloc(&_allocated_device_targets, required_target_bytes);
                } 
                else 
                {
                    _allocated_device_targets = nullptr; // Ensure it's null if not allocated
                }

                if (err_in != cudaSuccess || err_tgt != cudaSuccess) 
                {
                    fprintf(stderr, "[Batch %p] to_device(): ERROR - Failed to allocate CUDA memory (required size). Input err: %s, Target err: %s\n",
                           (void*)this, cudaGetErrorString(err_in), cudaGetErrorString(err_tgt)); fflush(stderr);
                    if (_allocated_device_inputs) safeCudaFree(&_allocated_device_inputs, "Batch alloc fail cleanup");
                    if (_allocated_device_targets) safeCudaFree(&_allocated_device_targets, "Batch alloc fail cleanup");
                    d_inputs = nullptr; d_targets = nullptr; is_data_on_device = false;
                    _allocated_input_bytes = 0; _allocated_target_bytes = 0;
                    return;
                }
                _allocated_input_bytes = required_input_bytes;
                _allocated_target_bytes = required_target_bytes;
            }

            size_t actual_input_bytes = inputs_flattened.size() * sizeof(float);
            size_t actual_target_bytes = targets_flattened.size() * sizeof(float);

            actual_input_bytes = std::min(actual_input_bytes, _allocated_input_bytes);
            actual_target_bytes = std::min(actual_target_bytes, _allocated_target_bytes);

            cudaError_t err_in = cudaMemcpy(_allocated_device_inputs, inputs_flattened.data(), actual_input_bytes, cudaMemcpyHostToDevice);
            cudaError_t err_tgt = cudaSuccess;
            if (actual_target_bytes > 0 && _allocated_device_targets) 
            {
                err_tgt = cudaMemcpy(_allocated_device_targets, targets_flattened.data(), actual_target_bytes, cudaMemcpyHostToDevice);
            }

            if (err_in != cudaSuccess || err_tgt != cudaSuccess) 
            {
                fprintf(stderr, "[Batch %p] to_device(): ERROR - Failed to copy actual data to CUDA device. Input err: %s, Target err: %s\n",
                       (void*)this, cudaGetErrorString(err_in), cudaGetErrorString(err_tgt)); fflush(stderr);
                // Don't free here, memory might still be usable next time, just mark as not ready
                d_inputs = nullptr; d_targets = nullptr; is_data_on_device = false;
                return;
            }

            d_inputs = _allocated_device_inputs;
            d_targets = _allocated_device_targets;
            is_data_on_device = true;

        } 
        else 
        {
          
            // --- CPU Path ---
            // If we were previously on CUDA, free the device memory as it's no longer needed
            if (is_data_on_device) 
            {
                 if (_allocated_device_inputs) safeCudaFree(&_allocated_device_inputs, "Batch internal d_inputs (CPU switch)");
                 if (_allocated_device_targets) safeCudaFree(&_allocated_device_targets, "Batch internal d_targets (CPU switch)");
                 _allocated_input_bytes = 0;
                 _allocated_target_bytes = 0;
            }

            d_inputs = inputs_flattened.empty() ? nullptr : inputs_flattened.data();
            d_targets = targets_flattened.empty() ? nullptr : targets_flattened.data();
            is_data_on_device = false;
        }
    }
};


class Dataset {
public:
    virtual ~Dataset() = default;
    virtual void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const = 0;
    virtual size_t size() const = 0;
    virtual size_t feature_size() const { return 0; }
    virtual size_t target_size() const { return 0; }
};


class DataLoader {
private:
    std::shared_ptr<Dataset> dataset;
    size_t batch_size;
    bool shuffle;
    size_t num_workers;
    size_t prefetch_factor;

    std::vector<size_t> indices;
    std::atomic<size_t> current_index{0};
    size_t num_batches = 0;

    std::queue<std::shared_ptr<Batch>> batch_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::vector<std::thread> workers;
    std::atomic<bool> stop_workers{false};
    std::atomic<size_t> workers_finished_count{0};

    std::string get_thread_id_str(std::thread::id id) 
    {
        std::stringstream ss;
        ss << id;
        return ss.str();
    }
    std::string get_thread_id_str() 
    {
        return get_thread_id_str(std::this_thread::get_id());
    }

    void worker_function() 
    {
        std::vector<float> sample_features;
        std::vector<float> sample_targets;
        size_t feat_size_hint = dataset->feature_size();
        size_t tgt_size_hint = dataset->target_size();
        if (feat_size_hint > 0) sample_features.reserve(feat_size_hint);
        if (tgt_size_hint > 0) sample_targets.reserve(tgt_size_hint);


        //SHITCODE BASED ON READABILITY, REFACTOR LATER, TOO MANY INDENTATIONS
        while (true) 
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this]{
                if (stop_workers.load()) return true;
                return (batch_queue.size() < prefetch_factor && current_index.load() < indices.size());
            });

            size_t current_idx_snapshot = current_index.load();
            if (stop_workers.load()) 
            {
                lock.unlock();
                workers_finished_count++;
                queue_cv.notify_all();
                return;
            }
            if (current_idx_snapshot >= indices.size()) 
            {
                 continue;
            }

            size_t start_idx = current_idx_snapshot;
            size_t end_idx = std::min(start_idx + this->batch_size, indices.size());
            if (!current_index.compare_exchange_strong(start_idx, end_idx)) 
            {
                 lock.unlock();
                 continue;
            }
            lock.unlock();

            size_t actual_batch_size = end_idx - start_idx;
            auto batch = std::make_shared<Batch>(actual_batch_size, this->batch_size, feat_size_hint, tgt_size_hint);

            if (feat_size_hint > 0) batch->inputs_flattened.resize(actual_batch_size * feat_size_hint);
            if (tgt_size_hint > 0) batch->targets_flattened.resize(actual_batch_size * tgt_size_hint);

            bool batch_ok = true;
            for (size_t i = 0; i < actual_batch_size; ++i) 
            {
                try 
                {
                    size_t data_idx = indices[start_idx + i];
                    dataset->get_sample(data_idx, sample_features, sample_targets);

                    size_t input_offset = i * feat_size_hint;
                    size_t target_offset = i * tgt_size_hint;

                    if (feat_size_hint > 0) 
                    {
                        if (sample_features.size() == feat_size_hint) 
                        {
                             std::memcpy(batch->inputs_flattened.data() + input_offset, sample_features.data(), feat_size_hint * sizeof(float));
                        } 
                        else 
                        {
                             fprintf(stderr, "DataLoader Worker: Warning - Mismatched feature size from get_sample (expected %zu, got %zu) for index %zu\n",
                                     feat_size_hint, sample_features.size(), data_idx); fflush(stderr);
                             batch_ok = false;
                        }
                    }
                    if (tgt_size_hint > 0 && !sample_targets.empty()) 
                    {
                        if (sample_targets.size() == tgt_size_hint) 
                        {
                             std::memcpy(batch->targets_flattened.data() + target_offset, sample_targets.data(), tgt_size_hint * sizeof(float));
                        } 
                        else 
                        {
                             fprintf(stderr, "DataLoader Worker: Warning - Mismatched target size from get_sample (expected %zu, got %zu) for index %zu\n",
                                     tgt_size_hint, sample_targets.size(), data_idx); fflush(stderr);
                             batch_ok = false;
                        }
                    } 
                    else if (tgt_size_hint > 0 && sample_targets.empty()) 
                    {
                         fprintf(stderr, "DataLoader Worker: Warning - Empty target vector from get_sample when target size hint is %zu for index %zu\n",
                                 tgt_size_hint, data_idx); fflush(stderr);
                         batch_ok = false;
                    }
                    if (!batch_ok) break;

                } 
                catch (const std::exception& e) 
                {
                    fprintf(stderr, "DataLoader Worker: Error getting sample %zu (original index %zu): %s\n", start_idx + i, indices[start_idx + i], e.what()); fflush(stderr);
                    batch->batch_size = 0; batch_ok = false; break;
                }
            }

            if (!batch_ok) 
            {
                batch->inputs_flattened.clear(); batch->targets_flattened.clear(); batch->batch_size = 0;
            }
            else 
            {
                 batch->to_device();
            }

            lock.lock();
            batch_queue.push(batch);
            lock.unlock();
            queue_cv.notify_one();
        }
    }

public:

    DataLoader(std::shared_ptr<Dataset> dataset,
               size_t batch_size = 1,
               bool shuffle = false,
               size_t num_workers = 0,
               size_t prefetch_factor = 2)
        : dataset(dataset),
          batch_size(batch_size),
          shuffle(shuffle),
          num_workers(num_workers),
          prefetch_factor(prefetch_factor > 0 ? prefetch_factor : 1)
          {
        if (!dataset) throw std::invalid_argument("Dataset cannot be null");
        if (batch_size == 0) throw std::invalid_argument("Batch size must be positive");
        if (dataset->size() == 0) fprintf(stderr, "Warning: DataLoader initialized with an empty dataset.\n"); fflush(stderr); 
        reset();
    }

    ~DataLoader() {
        stop_all_workers();
    }

private:
    void stop_all_workers() 
    {
         if (!workers.empty()) 
         {
             stop_workers = true;
             queue_cv.notify_all();
             for (size_t i = 0; i < workers.size(); ++i) 
             {
                 if (workers[i].joinable()) 
                 {
                     try 
                     {
                         workers[i].join();
                     } 
                     catch (const std::system_error& e) 
                     {
                          std::string worker_id_str = "N/A";
                          try { worker_id_str = get_thread_id_str(workers[i].get_id()); } catch(...) {}
                          fprintf(stderr, "[DataLoader %p] EXCEPTION joining worker %zu (ID: %s): %s\n", (void*)this, i, worker_id_str.c_str(), e.what()); fflush(stderr);
                     }
                 }
             }
             workers.clear();
             stop_workers = false;
             workers_finished_count = 0;
         }
    }

    void start_workers() 
    {
        if (num_workers > 0 && dataset->size() > 0) 
        {
             stop_workers = false;
             workers_finished_count = 0;
             workers.reserve(num_workers);
             for (size_t i = 0; i < num_workers; ++i) 
             {
                 workers.emplace_back(&DataLoader::worker_function, this);
             }
        }
    }

public:

    void reset() 
    {
        stop_all_workers();
        std::unique_lock<std::mutex> lock(queue_mutex);
        std::queue<std::shared_ptr<Batch>> empty_queue;
        std::swap(batch_queue, empty_queue);
        size_t ds_size = dataset ? dataset->size() : 0;
        indices.resize(ds_size);
        if (ds_size > 0) { std::iota(indices.begin(), indices.end(), 0); if (shuffle) { std::random_device rd; std::mt19937 g(rd()); std::shuffle(indices.begin(), indices.end(), g); } }
        current_index = 0;
        num_batches = (ds_size == 0 || this->batch_size == 0) ? 0 : (ds_size + this->batch_size - 1) / this->batch_size;
        workers_finished_count = 0;
        lock.unlock();
        start_workers();
    }

    std::shared_ptr<Batch> next_batch() 
    {
        if (num_workers == 0) 
        {

            size_t start_idx = current_index.load();
            if (start_idx >= indices.size()) return nullptr;
            size_t end_idx = std::min(start_idx + this->batch_size, indices.size());
            if (!current_index.compare_exchange_strong(start_idx, end_idx)) return nullptr;

            size_t actual_batch_size = end_idx - start_idx;
            size_t feat_size_hint = dataset->feature_size();
            size_t tgt_size_hint = dataset->target_size();
            auto batch = std::make_shared<Batch>(actual_batch_size, this->batch_size, feat_size_hint, tgt_size_hint);
            if (feat_size_hint > 0) batch->inputs_flattened.resize(actual_batch_size * feat_size_hint);
            if (tgt_size_hint > 0) batch->targets_flattened.resize(actual_batch_size * tgt_size_hint);

            std::vector<float> sample_features; std::vector<float> sample_targets;
            if (feat_size_hint > 0) sample_features.reserve(feat_size_hint);
            if (tgt_size_hint > 0) sample_targets.reserve(tgt_size_hint);

            bool batch_ok = true;
            for (size_t i = 0; i < actual_batch_size; ++i)
           {
                 try 
                 {
                    size_t data_idx = indices[start_idx + i];
                    dataset->get_sample(data_idx, sample_features, sample_targets);
      
                    if (!batch_ok) break;
                 } 
                 catch (const std::exception& e)
                 {
                     fprintf(stderr, "DataLoader Sync: Error getting sample %zu (original index %zu): %s\n", start_idx + i, indices[start_idx + i], e.what()); fflush(stderr);
                     return nullptr;
                 }
             }
             if (!batch_ok) 
             {
                 batch->batch_size = 0; batch->inputs_flattened.clear(); batch->targets_flattened.clear();
                 batch->to_device();
                 return batch;
             }
             batch->to_device();
             return batch;
        }

        // --- Async mode ---
        std::shared_ptr<Batch> batch = nullptr;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this]{
                return !batch_queue.empty() ||
                       (current_index.load() >= indices.size() && batch_queue.empty());
            });

            if (batch_queue.empty()) {
                 if (current_index.load() >= indices.size()) 
                 {
                     return nullptr;
                 } 
                 else 
                 {
              
                     fprintf(stderr, "[DataLoader %p] next_batch(): Async mode - WARNING: Woke up with empty queue but indices remain? Returning nullptr.\n", (void*)this); fflush(stderr);
                     return nullptr;
                 }
            }

            batch = batch_queue.front();
            batch_queue.pop();
        }
        queue_cv.notify_all();

        if (batch && batch->batch_size == 0) 
        {
             fprintf(stderr, "[DataLoader %p] next_batch(): Async mode - Retrieved an invalid batch (size 0). Skipping and getting next.\n", (void*)this); fflush(stderr);
             return next_batch();
        }
        return batch;
    }

    size_t get_num_batches() const 
    {
        return num_batches;
    }

    bool has_next() 
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return !batch_queue.empty() || current_index.load() < indices.size();
    }
};


class TensorDataset : public Dataset {
private:
    std::vector<std::vector<float>> inputs_host;
    std::vector<std::vector<float>> targets_host;
    size_t feature_size_ = 0;
    size_t target_size_ = 0;
public:
    TensorDataset(const std::vector<std::vector<float>>& inputs,
                  const std::vector<std::vector<float>>& targets)
        : inputs_host(inputs), targets_host(targets) {
        if (inputs_host.size() != targets_host.size()) throw std::invalid_argument("TensorDataset: Inputs and targets must have the same number of samples");
        if (!inputs_host.empty()) 
        {
            feature_size_ = inputs_host[0].size();
            if (!targets_host.empty() && !targets_host[0].empty()) 
            { 
                target_size_ = targets_host[0].size(); 
            } 
            else 
            { 
                target_size_ = 0; 
            }
        }
        else { fprintf(stderr, "Warning: TensorDataset created with zero samples.\n"); fflush(stderr); }
    }
    size_t size() const override { return inputs_host.size(); }
    void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const override 
    {
        if (index >= size()) 
        {
            fprintf(stderr, "Error: TensorDataset index %zu out of range (size %zu).\n", index, size()); fflush(stderr);
            throw std::out_of_range("Dataset index out of range");
        }
        out_features = inputs_host[index];
        if (!targets_host.empty()) 
        { 
            out_targets = targets_host[index]; 
        } 
        else 
        { 
            out_targets.clear(); 
        }
    }
    size_t feature_size() const override { return feature_size_; }
    size_t target_size() const override { return target_size_; }
};

#endif // DATALOADER_H