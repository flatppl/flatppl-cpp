cmake_minimum_required(VERSION 3.26)
project(flatppl_model LANGUAGES C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(FETCHCONTENT_FULLY_DISCONNECTED ON)
set(BUILD_SHARED_LIBS OFF)
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_BUILD_BINDINGS_TFLITE OFF)
set(IREE_ENABLE_CPUINFO OFF)
set(IREE_ENABLE_LIBBACKTRACE OFF)
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
add_subdirectory("${IREE_SOURCE}" iree EXCLUDE_FROM_ALL)

add_library(flatppl_generated_model SHARED
  src/model.c "${MODEL_SOURCE}/model_data.c")
target_compile_features(flatppl_generated_model PRIVATE c_std_11)
target_include_directories(flatppl_generated_model PRIVATE include "${MODEL_SOURCE}")
target_link_libraries(flatppl_generated_model PRIVATE iree::runtime::unified)
set_target_properties(flatppl_generated_model PROPERTIES
  OUTPUT_NAME "${MODEL_NAME}"
  C_VISIBILITY_PRESET hidden)
if(APPLE)
  target_link_options(flatppl_generated_model PRIVATE
    "LINKER:-exported_symbol,_flatppl_model_api_v1"
    "LINKER:-exported_symbol,_${MODEL_SYMBOL}"
    "LINKER:-dead_strip")
elseif(UNIX)
  target_link_options(flatppl_generated_model PRIVATE "LINKER:--exclude-libs,ALL")
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/artifact-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:flatppl_generated_model>\n")
if(WIN32)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/linker-$<CONFIG>.txt"
    CONTENT "$<TARGET_LINKER_FILE:flatppl_generated_model>\n")
endif()
