#ifndef FLATPPL_MODEL_HPP
#define FLATPPL_MODEL_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "flatppl.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace flatppl {

template <typename T, std::size_t Rank>
struct TensorView {
  const T* data;
  std::size_t size;
  std::array<std::uint64_t, Rank> shape;

  TensorView() = default;
  TensorView(const T* buffer, std::size_t count,
             std::array<std::uint64_t, Rank> dimensions)
      : data(buffer), size(count), shape(dimensions) {}

  template <std::size_t N, std::size_t R = Rank,
            std::enable_if_t<R == 1, int> = 0>
  TensorView(const T (&values)[N]) : TensorView(values, N, {N}) {}

  template <std::size_t N, std::size_t R = Rank,
            std::enable_if_t<R == 1, int> = 0>
  TensorView(const std::array<T, N>& values)
      : TensorView(values.data(), N, {N}) {}

  template <std::size_t R = Rank, std::enable_if_t<R == 1, int> = 0>
  TensorView(const std::vector<T>& values)
      : TensorView(values.data(), values.size(), {values.size()}) {}

  template <std::size_t Rows, std::size_t Columns, std::size_t R = Rank,
            std::enable_if_t<R == 2, int> = 0>
  TensorView(const T (&values)[Rows][Columns])
      : TensorView(values[0], Rows * Columns, {Rows, Columns}) {}

  flatppl_tensor descriptor() const {
    if (size > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
      throw std::length_error("FlatPPL input size overflows");
    }
    return {data, shape.data(), Rank, size * sizeof(T)};
  }
};

template <typename T, std::size_t Rank>
struct Tensor {
  std::vector<T> data;
  std::array<std::uint64_t, Rank> shape;
};

template <typename T>
flatppl_tensor scalar(const T& value) {
  return {&value, nullptr, 0, sizeof(T)};
}

template <typename T>
T read_scalar(const flatppl_tensor& value) {
  T result;
  std::memcpy(&result, value.data, sizeof(T));
  return result;
}

template <typename T, std::size_t Rank>
Tensor<T, Rank> read_tensor(const flatppl_tensor& value) {
  Tensor<T, Rank> result;
  std::copy_n(value.shape, Rank, result.shape.begin());
  result.data.resize(value.bytes / sizeof(T));
  if (value.bytes) {
    std::memcpy(result.data.data(), value.data, value.bytes);
  }
  return result;
}

class Library {
 public:
  explicit Library(const char* path) {
#ifdef _WIN32
    handle_ = LoadLibraryA(path);
    if (!handle_) throw std::runtime_error("could not load FlatPPL library");
#else
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle_) throw std::runtime_error(dlerror());
#endif
  }

  Library() = default;
  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  Library(Library&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}

  ~Library() {
    if (!handle_) return;
#ifdef _WIN32
    FreeLibrary(handle_);
#else
    dlclose(handle_);
#endif
  }

  const flatppl_model_api* api(const char* symbol) const {
#ifdef _WIN32
    auto get_api =
        reinterpret_cast<flatppl_get_api>(GetProcAddress(handle_, symbol));
#else
    auto get_api = reinterpret_cast<flatppl_get_api>(dlsym(handle_, symbol));
#endif
    if (!get_api) throw std::runtime_error("FlatPPL model symbol missing");
    return get_api();
  }

 private:
#ifdef _WIN32
  HMODULE handle_ = nullptr;
#else
  void* handle_ = nullptr;
#endif
};

class Instance {
 public:
  Instance(const flatppl_model_api* api, Library library)
      : library_(std::move(library)), api_(api) {
    if (!api_ || api_->version != 1)
      throw std::runtime_error("wrong FlatPPL ABI version");
    char error[512] = {};
    if (api_->create(&context_, error, sizeof(error)))
      throw std::runtime_error(error);
  }

  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  Instance(Instance&& other) noexcept
      : library_(std::move(other.library_)),
        api_(other.api_),
        context_(std::exchange(other.context_, nullptr)) {}

  ~Instance() {
    if (context_) api_->destroy(context_);
  }

  const flatppl_tensor* evaluate(const flatppl_tensor* inputs,
                                 std::size_t count) {
    const flatppl_tensor* outputs = nullptr;
    char error[512] = {};
    if (api_->evaluate(context_, inputs, count, &outputs, error,
                       sizeof(error))) {
      throw std::runtime_error(error);
    }
    return outputs;
  }

 private:
  Library library_;
  const flatppl_model_api* api_;
  void* context_ = nullptr;
};

}  // namespace flatppl
#endif
