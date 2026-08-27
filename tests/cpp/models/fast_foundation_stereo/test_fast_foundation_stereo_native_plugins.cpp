/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/fast_foundation_stereo/native_plugins/plugins.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using FullVolumePlugin = trtmc::FastFoundationStereoFullVolumeLeakyPlugin;
using Post8SumPlugin = trtmc::FastFoundationStereoPost8SumPlugin;

constexpr std::size_t kPositions = static_cast<std::size_t>(FullVolumePlugin::kBatch) *
                                   FullVolumePlugin::kDisparities * FullVolumePlugin::kHeight *
                                   FullVolumePlugin::kWidth;
constexpr std::size_t kPackedElements = kPositions * FullVolumePlugin::kChannelPitch;
constexpr std::size_t kLinearElements = kPositions * FullVolumePlugin::kChannels;
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::uint16_t kPositiveHalfBits = 0x3C3CU;
constexpr std::uint16_t kPost8SumHalfBits = 0x403CU;

static_assert(FullVolumePlugin::kBatch == Post8SumPlugin::kBatch);
static_assert(FullVolumePlugin::kChannels == Post8SumPlugin::kChannels);
static_assert(FullVolumePlugin::kDisparities == Post8SumPlugin::kDisparities);
static_assert(FullVolumePlugin::kHeight == Post8SumPlugin::kHeight);
static_assert(FullVolumePlugin::kWidth == Post8SumPlugin::kWidth);
static_assert(FullVolumePlugin::kChannelPitch == Post8SumPlugin::kChannelPitch);

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) {
        require_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr)
            cudaFree(pointer_);
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    void* get() const noexcept { return pointer_; }

  private:
    void* pointer_{nullptr};
};

class Stream {
  public:
    Stream() { require_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }

    ~Stream() {
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
    }

    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;

    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_{nullptr};
};

nvinfer1::PluginFieldCollection empty_fields() {
    nvinfer1::PluginFieldCollection fields{};
    fields.nbFields = 0;
    fields.fields = nullptr;
    return fields;
}

nvinfer1::PluginTensorDesc
make_desc(int32_t channels, nvinfer1::DataType type = nvinfer1::DataType::kHALF,
          nvinfer1::TensorFormat format = nvinfer1::TensorFormat::kDHWC8) {
    nvinfer1::PluginTensorDesc desc{};
    desc.dims.nbDims = 5;
    desc.dims.d[0] = FullVolumePlugin::kBatch;
    desc.dims.d[1] = channels;
    desc.dims.d[2] = FullVolumePlugin::kDisparities;
    desc.dims.d[3] = FullVolumePlugin::kHeight;
    desc.dims.d[4] = FullVolumePlugin::kWidth;
    desc.type = type;
    desc.format = format;
    return desc;
}

nvinfer1::DynamicPluginTensorDesc make_dynamic_desc(nvinfer1::PluginTensorDesc const& desc) {
    nvinfer1::DynamicPluginTensorDesc dynamic{};
    dynamic.desc = desc;
    dynamic.min = desc.dims;
    dynamic.max = desc.dims;
    return dynamic;
}

void require_full_volume_rejects(FullVolumePlugin& plugin,
                                 nvinfer1::PluginTensorDesc const& invalid,
                                 nvinfer1::PluginTensorDesc const& valid,
                                 const std::string& case_name) {
    require(plugin.onShapeChange(&invalid, 1, &valid, 1) != 0,
            "full-volume plugin accepted invalid input " + case_name);
    require(plugin.onShapeChange(&valid, 1, &invalid, 1) != 0,
            "full-volume plugin accepted invalid output " + case_name);
}

void test_full_volume_runtime_descriptors() {
    auto fields = empty_fields();
    FullVolumePlugin plugin(fields);
    auto logical = make_desc(FullVolumePlugin::kChannels);
    auto padded = make_desc(FullVolumePlugin::kChannelPitch);

    require(plugin.onShapeChange(&logical, 1, &logical, 1) == 0,
            "full-volume plugin rejected runtime C=28");
    require(plugin.onShapeChange(&padded, 1, &padded, 1) == 0,
            "full-volume plugin rejected TensorRT runtime C=32 padding");

    require_full_volume_rejects(plugin, make_desc(24), logical, "C=24");
    require_full_volume_rejects(plugin, make_desc(40), logical, "C=40");
    require_full_volume_rejects(plugin,
                                make_desc(FullVolumePlugin::kChannels, nvinfer1::DataType::kFLOAT),
                                logical, "dtype");
    require_full_volume_rejects(plugin,
                                make_desc(FullVolumePlugin::kChannels, nvinfer1::DataType::kHALF,
                                          nvinfer1::TensorFormat::kLINEAR),
                                logical, "format");
}

void test_full_volume_build_descriptors_remain_logical() {
    auto fields = empty_fields();
    FullVolumePlugin plugin(fields);
    auto logical = make_dynamic_desc(make_desc(FullVolumePlugin::kChannels));
    std::array<nvinfer1::DynamicPluginTensorDesc, 2> valid{logical, logical};

    require(plugin.supportsFormatCombination(0, valid.data(), 1, 1),
            "full-volume plugin rejected build-time input C=28");
    require(plugin.supportsFormatCombination(1, valid.data(), 1, 1),
            "full-volume plugin rejected build-time output C=28");
    require(plugin.configurePlugin(valid.data(), 1, valid.data() + 1, 1) == 0,
            "full-volume plugin rejected build-time profile C=28");

    auto padded = make_dynamic_desc(make_desc(FullVolumePlugin::kChannelPitch));
    std::array<nvinfer1::DynamicPluginTensorDesc, 2> invalid_input{padded, logical};
    std::array<nvinfer1::DynamicPluginTensorDesc, 2> invalid_output{logical, padded};
    require(!plugin.supportsFormatCombination(0, invalid_input.data(), 1, 1),
            "full-volume plugin accepted build-time input C=32");
    require(!plugin.supportsFormatCombination(1, invalid_output.data(), 1, 1),
            "full-volume plugin accepted build-time output C=32");
    require(plugin.configurePlugin(invalid_input.data(), 1, invalid_input.data() + 1, 1) != 0,
            "full-volume plugin accepted build-time input profile C=32");
    require(plugin.configurePlugin(invalid_output.data(), 1, invalid_output.data() + 1, 1) != 0,
            "full-volume plugin accepted build-time output profile C=32");
}

int32_t post8_shape_status(Post8SumPlugin& plugin, nvinfer1::PluginTensorDesc const& linear,
                           nvinfer1::PluginTensorDesc const& packed_input,
                           nvinfer1::PluginTensorDesc const& output) {
    std::array<nvinfer1::PluginTensorDesc, 2> inputs{linear, packed_input};
    return plugin.onShapeChange(inputs.data(), static_cast<int32_t>(inputs.size()), &output, 1);
}

void require_post8_rejects_packed(Post8SumPlugin& plugin, nvinfer1::PluginTensorDesc const& invalid,
                                  nvinfer1::PluginTensorDesc const& linear,
                                  nvinfer1::PluginTensorDesc const& valid,
                                  const std::string& case_name) {
    require(post8_shape_status(plugin, linear, invalid, valid) != 0,
            "post8-sum plugin accepted invalid packed input " + case_name);
    require(post8_shape_status(plugin, linear, valid, invalid) != 0,
            "post8-sum plugin accepted invalid output " + case_name);
}

void test_post8_runtime_descriptors() {
    auto fields = empty_fields();
    Post8SumPlugin plugin(fields);
    auto linear = make_desc(Post8SumPlugin::kChannels, nvinfer1::DataType::kHALF,
                            nvinfer1::TensorFormat::kLINEAR);
    auto logical = make_desc(Post8SumPlugin::kChannels);
    auto padded = make_desc(Post8SumPlugin::kChannelPitch);

    require(post8_shape_status(plugin, linear, logical, logical) == 0,
            "post8-sum plugin rejected runtime C=28");
    require(post8_shape_status(plugin, linear, padded, padded) == 0,
            "post8-sum plugin rejected TensorRT runtime C=32 padding");

    require_post8_rejects_packed(plugin, make_desc(24), linear, logical, "C=24");
    require_post8_rejects_packed(plugin, make_desc(40), linear, logical, "C=40");
    require_post8_rejects_packed(plugin,
                                 make_desc(Post8SumPlugin::kChannels, nvinfer1::DataType::kFLOAT),
                                 linear, logical, "dtype");
    require_post8_rejects_packed(plugin,
                                 make_desc(Post8SumPlugin::kChannels, nvinfer1::DataType::kHALF,
                                           nvinfer1::TensorFormat::kLINEAR),
                                 linear, logical, "format");

    auto wrong_linear_dtype = linear;
    wrong_linear_dtype.type = nvinfer1::DataType::kFLOAT;
    require(post8_shape_status(plugin, wrong_linear_dtype, logical, logical) != 0,
            "post8-sum plugin accepted invalid linear input dtype");
    auto wrong_linear_format = linear;
    wrong_linear_format.format = nvinfer1::TensorFormat::kDHWC8;
    require(post8_shape_status(plugin, wrong_linear_format, logical, logical) != 0,
            "post8-sum plugin accepted invalid linear input format");
}

void test_post8_build_descriptors_remain_logical() {
    auto fields = empty_fields();
    Post8SumPlugin plugin(fields);
    auto linear = make_dynamic_desc(make_desc(Post8SumPlugin::kChannels, nvinfer1::DataType::kHALF,
                                              nvinfer1::TensorFormat::kLINEAR));
    auto logical = make_dynamic_desc(make_desc(Post8SumPlugin::kChannels));
    std::array<nvinfer1::DynamicPluginTensorDesc, 3> valid{linear, logical, logical};

    for (int32_t position = 0; position < static_cast<int32_t>(valid.size()); ++position) {
        require(plugin.supportsFormatCombination(position, valid.data(), 2, 1),
                "post8-sum plugin rejected build-time C=28 descriptor");
    }
    require(plugin.configurePlugin(valid.data(), 2, valid.data() + 2, 1) == 0,
            "post8-sum plugin rejected build-time profile C=28");

    auto padded = make_dynamic_desc(make_desc(Post8SumPlugin::kChannelPitch));
    std::array<nvinfer1::DynamicPluginTensorDesc, 3> invalid_input{linear, padded, logical};
    std::array<nvinfer1::DynamicPluginTensorDesc, 3> invalid_output{linear, logical, padded};
    require(!plugin.supportsFormatCombination(1, invalid_input.data(), 2, 1),
            "post8-sum plugin accepted build-time packed input C=32");
    require(!plugin.supportsFormatCombination(2, invalid_output.data(), 2, 1),
            "post8-sum plugin accepted build-time output C=32");
    require(plugin.configurePlugin(invalid_input.data(), 2, invalid_input.data() + 2, 1) != 0,
            "post8-sum plugin accepted build-time input profile C=32");
    require(plugin.configurePlugin(invalid_output.data(), 2, invalid_output.data() + 2, 1) != 0,
            "post8-sum plugin accepted build-time output profile C=32");
}

void require_padded_lanes_are_zero(void* device_output, const std::string& plugin_name) {
    constexpr std::size_t row_pitch = FullVolumePlugin::kChannelPitch * kHalfBytes;
    constexpr std::size_t padding_bytes =
        (FullVolumePlugin::kChannelPitch - FullVolumePlugin::kChannels) * kHalfBytes;
    constexpr std::size_t padding_lanes =
        FullVolumePlugin::kChannelPitch - FullVolumePlugin::kChannels;
    std::vector<std::uint16_t> padding(kPositions * padding_lanes);
    auto* first_padding_lane =
        static_cast<std::uint8_t*>(device_output) + FullVolumePlugin::kChannels * kHalfBytes;
    require_cuda(cudaMemcpy2D(padding.data(), padding_bytes, first_padding_lane, row_pitch,
                              padding_bytes, kPositions, cudaMemcpyDeviceToHost),
                 "cudaMemcpy2D padded lanes");
    require(
        std::all_of(padding.begin(), padding.end(), [](std::uint16_t bits) { return bits == 0U; }),
        plugin_name + " left a non-zero padded lane");
}

void test_full_volume_enqueue_zeros_padded_lanes() {
    auto fields = empty_fields();
    FullVolumePlugin plugin(fields);
    auto logical = make_desc(FullVolumePlugin::kChannels);
    auto padded = make_desc(FullVolumePlugin::kChannelPitch);
    std::array<nvinfer1::PluginTensorDesc, 2> runtime_descs{logical, padded};
    DeviceBuffer input(kPackedElements * kHalfBytes);
    DeviceBuffer output(kPackedElements * kHalfBytes);
    Stream stream;

    require_cuda(cudaMemset(input.get(), 0x3C, kPackedElements * kHalfBytes),
                 "cudaMemset full-volume input");
    void const* inputs[]{input.get()};
    void* outputs[]{output.get()};
    for (auto const& desc : runtime_descs) {
        std::string case_name = "runtime C=" + std::to_string(desc.dims.d[1]);
        require_cuda(cudaMemset(output.get(), 0x7F, kPackedElements * kHalfBytes),
                     "cudaMemset full-volume output");
        require(plugin.enqueue(&desc, &desc, inputs, outputs, nullptr, stream.get()) == 0,
                "full-volume plugin enqueue failed for " + case_name);
        require_cuda(cudaStreamSynchronize(stream.get()), "full-volume kernel synchronization");

        std::uint16_t logical_output = 0U;
        require_cuda(cudaMemcpy(&logical_output, output.get(), sizeof(logical_output),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy full-volume logical output");
        require(logical_output == kPositiveHalfBits,
                "full-volume kernel did not preserve a positive logical lane for " + case_name);
        require_padded_lanes_are_zero(output.get(), "full-volume plugin " + case_name);
    }
}

void test_post8_enqueue_zeros_padded_lanes() {
    auto fields = empty_fields();
    Post8SumPlugin plugin(fields);
    auto linear_desc = make_desc(Post8SumPlugin::kChannels, nvinfer1::DataType::kHALF,
                                 nvinfer1::TensorFormat::kLINEAR);
    auto logical_desc = make_desc(Post8SumPlugin::kChannels);
    auto padded_desc = make_desc(Post8SumPlugin::kChannelPitch);
    std::array<nvinfer1::PluginTensorDesc, 2> runtime_packed_descs{logical_desc, padded_desc};
    DeviceBuffer linear(kLinearElements * kHalfBytes);
    DeviceBuffer skip(kPackedElements * kHalfBytes);
    DeviceBuffer output(kPackedElements * kHalfBytes);
    Stream stream;

    require_cuda(cudaMemset(linear.get(), 0x3C, kLinearElements * kHalfBytes),
                 "cudaMemset post8 linear input");
    require_cuda(cudaMemset(skip.get(), 0x3C, kPackedElements * kHalfBytes),
                 "cudaMemset post8 packed input");
    void const* inputs[]{linear.get(), skip.get()};
    void* outputs[]{output.get()};
    for (auto const& packed_desc : runtime_packed_descs) {
        std::array<nvinfer1::PluginTensorDesc, 2> input_descs{linear_desc, packed_desc};
        std::string case_name = "runtime C=" + std::to_string(packed_desc.dims.d[1]);
        require_cuda(cudaMemset(output.get(), 0x7F, kPackedElements * kHalfBytes),
                     "cudaMemset post8 output");
        require(plugin.enqueue(input_descs.data(), &packed_desc, inputs, outputs, nullptr,
                               stream.get()) == 0,
                "post8-sum plugin enqueue failed for " + case_name);
        require_cuda(cudaStreamSynchronize(stream.get()), "post8-sum kernel synchronization");

        std::uint16_t logical_output = 0U;
        require_cuda(cudaMemcpy(&logical_output, output.get(), sizeof(logical_output),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy post8 logical output");
        require(logical_output == kPost8SumHalfBits,
                "post8-sum kernel produced an unexpected logical value for " + case_name);
        require_padded_lanes_are_zero(output.get(), "post8-sum plugin " + case_name);
    }
}

bool gpu_available() {
    int32_t device_count = 0;
    cudaError_t result = cudaGetDeviceCount(&device_count);
    if (result == cudaSuccess) {
        if (device_count > 0)
            return true;
        std::cerr << "SKIP: no CUDA device available\n";
        return false;
    }
    if (result == cudaErrorNoDevice) {
        cudaGetLastError();
        std::cerr << "SKIP: no CUDA device available\n";
        return false;
    }
    require_cuda(result, "cudaGetDeviceCount");
    return false; // Unreachable, keeps all compiler control-flow analyses satisfied.
}

} // namespace

int main() {
    try {
        test_full_volume_runtime_descriptors();
        test_full_volume_build_descriptors_remain_logical();
        test_post8_runtime_descriptors();
        test_post8_build_descriptors_remain_logical();
        if (!gpu_available())
            return 0;
        test_full_volume_enqueue_zeros_padded_lanes();
        test_post8_enqueue_zeros_padded_lanes();
        // Temporary GPU-only probe for PR #1056. Remove after validating external log propagation.
        require(false, "INTENTIONAL_INTERNAL_CI_GPU_LOG_PROBE: verifying GPU failure logs are "
                       "visible to external PR authors");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
