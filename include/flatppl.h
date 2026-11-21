#ifndef FLATPPL_MODEL_H
#define FLATPPL_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t flatppl_type;
enum {
  FLATPPL_F32,
  FLATPPL_F64,
  FLATPPL_I32,
  FLATPPL_I64,
  FLATPPL_U64,
  FLATPPL_BOOL,
};

typedef struct {
  const char* name;
  flatppl_type type;
  size_t rank;
  const int64_t* shape; /* -1 marks a dynamic dimension. */
} flatppl_field;

typedef struct {
  const void* data;
  const uint64_t* shape;
  size_t rank;
  size_t bytes;
} flatppl_tensor;

typedef struct {
  uint32_t version;
  size_t input_count;
  const flatppl_field* inputs;
  size_t output_count;
  const flatppl_field* outputs;
  /* Return zero on success. Each context supports one call at a time. */
  int (*create)(void** context, char* error, size_t error_capacity);
  /* Results belong to the context and expire at the next call or destruction.
   */
  int (*evaluate)(void* context, const flatppl_tensor* inputs, size_t count,
                  const flatppl_tensor** outputs, char* error,
                  size_t error_capacity);
  void (*destroy)(void* context);
} flatppl_model_api;

typedef const flatppl_model_api* (*flatppl_get_api)(void);

/* Look this symbol up on the model library handle for generic loading. */
const flatppl_model_api* flatppl_model_api_v1(void);

#ifdef __cplusplus
}
#endif
#endif
