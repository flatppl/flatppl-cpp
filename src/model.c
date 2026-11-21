#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "flatppl.h"
#include "iree/runtime/api.h"
#include "model_config.h"

extern const unsigned char flatppl_module[];
extern const size_t flatppl_module_size;

typedef struct {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
  iree_runtime_call_t call;
  int call_initialized;
  flatppl_tensor results[OUTPUT_COUNT ? OUTPUT_COUNT : 1];
} model_context;

static void clear_results(model_context* model) {
  for (size_t i = 0; i < OUTPUT_COUNT; ++i) {
    free((void*)model->results[i].data);
    free((void*)model->results[i].shape);
    model->results[i] = (flatppl_tensor){0};
  }
}

static void model_destroy(void* context) {
  model_context* model = context;
  if (!model) return;
  clear_results(model);
  if (model->call_initialized) iree_runtime_call_deinitialize(&model->call);
  iree_runtime_session_release(model->session);
  iree_hal_device_release(model->device);
  iree_runtime_instance_release(model->instance);
  free(model);
}

static int report(iree_status_t status, char* error, size_t capacity) {
  if (iree_status_is_ok(status)) return 0;
  if (error && capacity) {
    iree_host_size_t used = 0;
    if (!iree_status_format(status, capacity, error, &used)) {
      snprintf(error, capacity, "%s",
               iree_status_code_string(iree_status_code(status)));
    }
  }
  iree_status_ignore(status);
  return 1;
}

static iree_status_t initialize(model_context* model) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &instance_options, allocator, &model->instance));
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      model->instance, iree_make_cstring_view("local-sync"), &model->device));

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      model->instance, &session_options, model->device, allocator,
      &model->session));
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_memory(
      model->session,
      iree_make_const_byte_span(flatppl_module, flatppl_module_size),
      iree_allocator_null()));
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      model->session, iree_make_cstring_view(MODEL_ENTRY_POINT), &model->call));
  model->call_initialized = 1;
  return iree_ok_status();
}

static int model_create(void** context, char* error, size_t capacity) {
  if (!context)
    return report(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                   "missing context pointer"),
                  error, capacity);
  *context = NULL;
  model_context* model = calloc(1, sizeof(*model));
  if (!model)
    return report(iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                   "could not allocate context"),
                  error, capacity);
  if (report(initialize(model), error, capacity)) {
    model_destroy(model);
    return 1;
  }
  *context = model;
  return 0;
}

typedef struct {
  iree_hal_element_type_t iree_type;
  size_t bytes;
} element_type;

static const element_type elements[] = {
    [FLATPPL_F32] = {IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4},
    [FLATPPL_F64] = {IREE_HAL_ELEMENT_TYPE_FLOAT_64, 8},
    [FLATPPL_I32] = {IREE_HAL_ELEMENT_TYPE_SINT_32, 4},
    [FLATPPL_I64] = {IREE_HAL_ELEMENT_TYPE_SINT_64, 8},
    [FLATPPL_U64] = {IREE_HAL_ELEMENT_TYPE_UINT_64, 8},
    [FLATPPL_BOOL] = {IREE_HAL_ELEMENT_TYPE_BOOL_8, 1},
};

static iree_status_t validate(const flatppl_field* field,
                              const flatppl_tensor* value) {
  if (value->rank != field->rank || (value->rank && !value->shape)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s: wrong tensor rank", field->name);
  }
  size_t bytes = elements[field->type].bytes;
  for (size_t axis = 0; axis < value->rank; ++axis) {
    uint64_t length = value->shape[axis];
    if (length > INT64_MAX ||
        (field->shape[axis] >= 0 && length != (uint64_t)field->shape[axis])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s: wrong dimension", field->name);
    }
    if (length && bytes > SIZE_MAX / length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s: size overflows", field->name);
    }
    bytes *= (size_t)length;
  }
  if (bytes != value->bytes || (bytes && !value->data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s: wrong buffer size", field->name);
  }
  return iree_ok_status();
}

static int64_t read_integer(const void* data, size_t bytes) {
  if (bytes == sizeof(int32_t)) {
    int32_t value;
    memcpy(&value, data, sizeof(value));
    return value;
  }
  int64_t value;
  memcpy(&value, data, sizeof(value));
  return value;
}

static iree_status_t push_input(model_context* model,
                                const flatppl_tensor* value,
                                flatppl_type type) {
  iree_hal_dim_t shape[MAX_RANK];
  for (size_t axis = 0; axis < value->rank; ++axis) {
    shape[axis] = (iree_hal_dim_t)value->shape[axis];
  }
  iree_hal_buffer_view_t* view = NULL;
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      model->device, iree_runtime_session_device_allocator(model->session),
      value->rank, shape, elements[type].iree_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(value->data, value->bytes), &view));
  iree_status_t status =
      iree_runtime_call_inputs_push_back_buffer_view(&model->call, view);
  iree_hal_buffer_view_release(view);
  return status;
}

static iree_status_t copy_result(model_context* model, size_t index,
                                 iree_hal_buffer_view_t* view) {
  size_t rank = iree_hal_buffer_view_shape_rank(view);
  iree_device_size_t size = iree_hal_buffer_view_byte_length(view);
  if (rank != output_fields[index].rank || size > SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "unexpected FlatPPL output shape");
  }
  size_t bytes = (size_t)size;
  uint64_t* shape = rank ? malloc(rank * sizeof(*shape)) : NULL;
  void* data = bytes ? malloc(bytes) : NULL;
  model->results[index] = (flatppl_tensor){data, shape, rank, bytes};
  if ((rank && !shape) || (bytes && !data)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "could not allocate result");
  }
  const iree_hal_dim_t* dimensions = iree_hal_buffer_view_shape_dims(view);
  for (size_t axis = 0; axis < rank; ++axis) shape[axis] = dimensions[axis];
  IREE_RETURN_IF_ERROR(validate(&output_fields[index], &model->results[index]));
  if (!bytes) return iree_ok_status();
  return iree_hal_device_transfer_d2h(
      model->device, iree_hal_buffer_view_buffer(view), 0, data, bytes,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

static iree_status_t check_model_status(model_context* model) {
  if (MODEL_STATUS_BYTES == 0) return iree_ok_status();
  iree_hal_buffer_view_t* view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&model->call, &view));
  unsigned char bytes[sizeof(int64_t)];
  iree_status_t transfer = iree_hal_device_transfer_d2h(
      model->device, iree_hal_buffer_view_buffer(view), 0, bytes,
      MODEL_STATUS_BYTES, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout());
  iree_hal_buffer_view_release(view);
  IREE_RETURN_IF_ERROR(transfer);
  int64_t code = read_integer(bytes, MODEL_STATUS_BYTES);
  return code ? iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                 "FlatPPL model status %lld", (long long)code)
              : iree_ok_status();
}

static iree_status_t evaluate(model_context* model,
                              const flatppl_tensor* inputs) {
  for (size_t i = 0; i < INPUT_COUNT; ++i) {
    IREE_RETURN_IF_ERROR(validate(&input_fields[i], &inputs[i]));
  }
  for (size_t i = 0; i < INPUT_COUNT; ++i) {
    int count_input = leading_extent_inputs[i];
    if (count_input >= 0) {
      const flatppl_tensor* count = &inputs[count_input];
      int64_t rows = read_integer(count->data, count->bytes);
      if (rows < 0 || (uint64_t)rows != inputs[i].shape[0]) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT, "%s: dimension disagrees with %s",
            input_fields[i].name, input_fields[count_input].name);
      }
    }
    IREE_RETURN_IF_ERROR(push_input(model, &inputs[i], input_fields[i].type));
  }
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&model->call, 0));
  for (size_t i = 0; i < OUTPUT_COUNT; ++i) {
    iree_hal_buffer_view_t* view = NULL;
    IREE_RETURN_IF_ERROR(
        iree_runtime_call_outputs_pop_front_buffer_view(&model->call, &view));
    iree_status_t status = copy_result(model, i, view);
    iree_hal_buffer_view_release(view);
    IREE_RETURN_IF_ERROR(status);
  }
  return check_model_status(model);
}

static int model_evaluate(void* context, const flatppl_tensor* inputs,
                          size_t count, const flatppl_tensor** outputs,
                          char* error, size_t capacity) {
  if (!outputs)
    return report(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                   "missing outputs pointer"),
                  error, capacity);
  *outputs = NULL;
  model_context* model = context;
  if (!model)
    return report(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing context"),
        error, capacity);
  clear_results(model);
  if (count != INPUT_COUNT || (count && !inputs)) {
    return report(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "wrong input count"),
        error, capacity);
  }
  iree_status_t status = evaluate(model, inputs);
  iree_runtime_call_reset(&model->call);
  if (report(status, error, capacity)) {
    clear_results(model);
    return 1;
  }
  *outputs = model->results;
  return 0;
}

static const flatppl_model_api api = {
    1,
    INPUT_COUNT,
    input_fields,
    OUTPUT_COUNT,
    output_fields,
    model_create,
    model_evaluate,
    model_destroy,
};

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT const flatppl_model_api* MODEL_SYMBOL(void) { return &api; }
EXPORT const flatppl_model_api* flatppl_model_api_v1(void) { return &api; }
