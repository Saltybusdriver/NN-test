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
#include <deque>
#include <chrono>

class Dataset;

template <typename T>
struct CudaAllocator {
    using value_type = T;
    CudaAllocator() = default;
    template <typename U> constexpr CudaAllocator(const CudaAllocator<U>&) noexcept {}
    
    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        T* p = nullptr;
        cudaError_t err = cudaMallocHost((void**)&p, n * sizeof(T));
        if (err != cudaSuccess) throw std::bad_alloc();
        return p;
    }
    
    void deallocate(T* p, std::size_t n) noexcept {
        if (p) cudaFreeHost(p);
    }
};

template <typename T, typename U> bool operator==(const CudaAllocator<T>&, const CudaAllocator<U>&) { return true; }
template <typename T, typename U> bool operator!=(const CudaAllocator<T>&, const CudaAllocator<U>&) { return false; }

struct Batch {
    std::vector<float, CudaAllocator<float>> inputs_flattened;
    std::vector<float, CudaAllocator<float>> targets_flattened;
    size_t batch_size = 0; 

    size_t max_batch_size_ = 0;
    size_t feature_size_hint_ = 0;
    size_t target_size_hint_ = 0;

    float* d_inputs = nullptr;  
    float* d_targets = nullptr; 
    bool is_data_on_device = false;
    cudaEvent_t copy_event = nullptr;
    cudaEvent_t compute_event = nullptr;

private:
    float* _allocated_device_inputs = nullptr;
    float* _allocated_device_targets = nullptr;

public:
    Batch(size_t actual_size, size_t max_size, size_t feat_hint, size_t tgt_hint)
        : batch_size(actual_size), max_batch_size_(max_size),
          feature_size_hint_(feat_hint), target_size_hint_(tgt_hint) 
    {
        inputs_flattened.resize(max_size * feat_hint);
        if (tgt_hint > 0) targets_flattened.resize(max_size * tgt_hint);

        cudaMalloc(&_allocated_device_inputs, max_size * feat_hint * sizeof(float));
        if (tgt_hint > 0) cudaMalloc(&_allocated_device_targets, max_size * tgt_hint * sizeof(float));
        
        cudaEventCreateWithFlags(&copy_event, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming);
    }

    Batch() = default;

    ~Batch() {
        if (_allocated_device_inputs) cudaFree(_allocated_device_inputs);
        if (_allocated_device_targets) cudaFree(_allocated_device_targets);
        if (copy_event) cudaEventDestroy(copy_event);
        if (compute_event) cudaEventDestroy(compute_event);
    }

    void to_device(cudaStream_t stream = 0) {
        if (batch_size == 0) return;
        d_inputs = _allocated_device_inputs;
        d_targets = _allocated_device_targets;

        cudaMemcpyAsync(d_inputs, inputs_flattened.data(), 
                        batch_size * feature_size_hint_ * sizeof(float), 
                        cudaMemcpyHostToDevice, stream);
        
        if (target_size_hint_ > 0 && d_targets) {
            cudaMemcpyAsync(d_targets, targets_flattened.data(), 
                            batch_size * target_size_hint_ * sizeof(float), 
                            cudaMemcpyHostToDevice, stream);
        }

        cudaEventRecord(copy_event, stream);
        is_data_on_device = true;
    }
};

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const = 0;
    virtual size_t size() const = 0;
    virtual size_t feature_size() const = 0;
    virtual size_t target_size() const = 0;

    virtual void copy_sample_to(size_t index, float* out_features, float* out_targets) const {
        std::vector<float> features;
        std::vector<float> targets;
        get_sample(index, features, targets);

        if (out_features && !features.empty()) {
            std::memcpy(out_features, features.data(), features.size() * sizeof(float));
        }
        if (out_targets && !targets.empty()) {
            std::memcpy(out_targets, targets.data(), targets.size() * sizeof(float));
        }
    }
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
    std::queue<std::shared_ptr<Batch>> upload_queue;
    std::queue<std::shared_ptr<Batch>> empty_queue; 
    std::deque<std::shared_ptr<Batch>> pending_recycle;
    std::shared_ptr<Batch> last_returned_batch = nullptr;
    size_t active_fill_count = 0;
    size_t active_upload_count = 0;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::vector<std::thread> workers;
    std::thread uploader;
    std::atomic<bool> stop_workers{false};
    
    cudaStream_t data_stream;

    void recycle_completed_batches_locked() {
        auto it = pending_recycle.begin();
        while (it != pending_recycle.end()) {
            const auto& batch = *it;
            cudaError_t ready = cudaEventQuery(batch->compute_event);
            if (ready == cudaSuccess) {
                empty_queue.push(batch);
                it = pending_recycle.erase(it);
            } else if (ready == cudaErrorNotReady) {
                ++it;
            } else {
                cudaGetLastError();
                empty_queue.push(batch);
                it = pending_recycle.erase(it);
            }
        }
    }

    bool all_batches_assigned_locked() const {
        return current_index.load() >= indices.size();
    }

    bool loading_finished_locked() const {
        return all_batches_assigned_locked() &&
               active_fill_count == 0 &&
               active_upload_count == 0 &&
               upload_queue.empty() &&
               batch_queue.empty();
    }

    void worker_function() {
        size_t feat_size = dataset->feature_size();
        size_t tgt_size = dataset->target_size();

        while (true) {
            std::shared_ptr<Batch> batch;
            size_t start_idx;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] {
                    return stop_workers || (!empty_queue.empty() && current_index.load() < indices.size());
                });

                if (stop_workers) return;

                start_idx = current_index.load();
                if (start_idx >= indices.size()) {
                    queue_cv.notify_all();
                    return;
                }

                batch = empty_queue.front();
                empty_queue.pop();
                
                size_t end_idx = std::min(start_idx + batch_size, indices.size());
                current_index.store(end_idx);
                batch->batch_size = end_idx - start_idx;
                active_fill_count++;
            }

            for (size_t i = 0; i < batch->batch_size; ++i) {
                float* features_out = batch->inputs_flattened.data() + (i * feat_size);
                float* targets_out = (tgt_size > 0) ? batch->targets_flattened.data() + (i * tgt_size) : nullptr;
                dataset->copy_sample_to(indices[start_idx + i], features_out, targets_out);
            }

            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                active_fill_count--;
                upload_queue.push(batch);
            }
            queue_cv.notify_all();
        }
    }

    void uploader_function() {
        while (true) {
            std::shared_ptr<Batch> batch;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                recycle_completed_batches_locked();
                queue_cv.notify_all();

                queue_cv.wait_for(lock, std::chrono::milliseconds(1), [this] {
                    return stop_workers ||
                           !upload_queue.empty();
                });

                recycle_completed_batches_locked();
                queue_cv.notify_all();

                if (upload_queue.empty()) {
                    if (stop_workers && active_fill_count == 0) return;
                    continue;
                }

                batch = upload_queue.front();
                upload_queue.pop();
                active_upload_count++;
            }

            batch->to_device(data_stream);

            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                active_upload_count--;
                batch_queue.push(batch);
            }
            queue_cv.notify_all();
        }
    }

public:
    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size = 1, bool shuffle = false, size_t num_workers = 4, size_t prefetch_factor = 2)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), num_workers(num_workers), prefetch_factor(prefetch_factor) 
    {
        // Non-blocking stream prevents the loader from syncing with the main compute stream
        cudaStreamCreateWithFlags(&data_stream, cudaStreamNonBlocking);
        reset();
    }

    ~DataLoader() {
        stop_all_workers();
        cudaStreamDestroy(data_stream);
    }

    void reset() {
        stop_all_workers();
        std::lock_guard<std::mutex> lock(queue_mutex);
        
        while (!batch_queue.empty()) { empty_queue.push(batch_queue.front()); batch_queue.pop(); }
        while (!upload_queue.empty()) { empty_queue.push(upload_queue.front()); upload_queue.pop(); }
        while (!pending_recycle.empty()) {
            auto batch = pending_recycle.front();
            pending_recycle.pop_front();
            if (batch->is_data_on_device && batch->compute_event) {
                cudaEventSynchronize(batch->compute_event);
            }
            empty_queue.push(batch);
        }
        if (last_returned_batch) {
            if (last_returned_batch->is_data_on_device && last_returned_batch->compute_event) {
                cudaEventSynchronize(last_returned_batch->compute_event);
            }
            empty_queue.push(last_returned_batch);
            last_returned_batch = nullptr;
        }
        active_fill_count = 0;
        active_upload_count = 0;

        // Pre-fill pool if empty (only on first start or if batch size changed)
        size_t required_pool = num_workers + (prefetch_factor * num_workers) + 2;
        while (empty_queue.size() < required_pool) {
            empty_queue.push(std::make_shared<Batch>(0, batch_size, dataset->feature_size(), dataset->target_size()));
        }

        indices.resize(dataset->size());
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle) std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

        current_index = 0;
        num_batches = (dataset->size() + batch_size - 1) / batch_size;
        
        stop_workers = false;
        for (size_t i = 0; i < num_workers; ++i) {
            workers.emplace_back(&DataLoader::worker_function, this);
        }
        uploader = std::thread(&DataLoader::uploader_function, this);
        queue_cv.notify_all();
    }

    std::shared_ptr<Batch> next_batch(cudaStream_t compute_stream = 0) {
        if (last_returned_batch) {
            if (last_returned_batch->compute_event) { 
                cudaEventRecord(last_returned_batch->compute_event, compute_stream);
            }

            std::lock_guard<std::mutex> lock(queue_mutex);
            pending_recycle.push_back(last_returned_batch);
            last_returned_batch = nullptr;
            recycle_completed_batches_locked();
            queue_cv.notify_all();
        }

        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [this] { 
            return !batch_queue.empty() || loading_finished_locked(); 
        });

        if (batch_queue.empty()) return nullptr;

        auto batch = batch_queue.front();
        batch_queue.pop();
        last_returned_batch = batch;
        lock.unlock();

        // GPU compute stream waits ONLY for the specific event of this batch
        if (batch->is_data_on_device && batch->copy_event) {
            cudaStreamWaitEvent(compute_stream, batch->copy_event, 0);
        }

        return batch;
    }

    size_t get_num_batches() const { return num_batches; }
    bool has_next() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return !batch_queue.empty() ||
               !upload_queue.empty() ||
               active_fill_count > 0 ||
               active_upload_count > 0 ||
               current_index < indices.size();
    }

private:
    void stop_all_workers() {
        stop_workers = true;
        queue_cv.notify_all();
        for (auto& t : workers) if (t.joinable()) t.join();
        workers.clear();
        queue_cv.notify_all();
        if (uploader.joinable()) uploader.join();
    }
};

class TensorDataset : public Dataset {
private:
    std::vector<std::vector<float>> inputs_host;
    std::vector<std::vector<float>> targets_host;
    size_t feature_size_ = 0;
    size_t target_size_ = 0;
public:
    TensorDataset(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets)
        : inputs_host(inputs), targets_host(targets) {
        if (!inputs_host.empty()) {
            feature_size_ = inputs_host[0].size();
            target_size_ = (!targets_host.empty()) ? targets_host[0].size() : 0;
        }
    }
    size_t size() const override { return inputs_host.size(); }
    void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const override {
        out_features = inputs_host[index];
        if (!targets_host.empty()) out_targets = targets_host[index];
    }
    size_t feature_size() const override { return feature_size_; }
    size_t target_size() const override { return target_size_; }

    void copy_sample_to(size_t index, float* out_features, float* out_targets) const override {
        if (out_features && feature_size_ > 0) {
            std::memcpy(out_features, inputs_host[index].data(), feature_size_ * sizeof(float));
        }
        if (out_targets && target_size_ > 0 && !targets_host.empty()) {
            std::memcpy(out_targets, targets_host[index].data(), target_size_ * sizeof(float));
        }
    }
};

#endif
