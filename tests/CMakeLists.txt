cmake_minimum_required(VERSION 3.26)
project(flatppl_cpp_tests LANGUAGES CXX)
enable_testing()

set(MODEL_NAME normal CACHE STRING "FlatPPL example to test")
find_file(MODEL_LIBRARY
  NAMES "${CMAKE_SHARED_LIBRARY_PREFIX}${MODEL_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
  PATHS "${MODEL_DIR}" NO_DEFAULT_PATH REQUIRED NO_CACHE)

add_library(model_library SHARED IMPORTED)
set_target_properties(model_library PROPERTIES IMPORTED_LOCATION "${MODEL_LIBRARY}")
if(WIN32)
  find_library(MODEL_IMPLIB NAMES "${MODEL_NAME}" PATHS "${MODEL_DIR}"
    NO_DEFAULT_PATH REQUIRED NO_CACHE)
  set_target_properties(model_library PROPERTIES IMPORTED_IMPLIB "${MODEL_IMPLIB}")
endif()

foreach(mode linked open)
  add_executable("model_${mode}" "${MODEL_NAME}.cc")
  target_include_directories("model_${mode}" PRIVATE "${MODEL_DIR}")
  set_target_properties("model_${mode}" PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options("model_${mode}" PRIVATE -Wall -Wextra -Werror)
  endif()
endforeach()

target_compile_definitions(model_linked PRIVATE FLATPPL_TEST_LINKED)
target_link_libraries(model_linked PRIVATE model_library)
target_link_libraries(model_open PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME "${MODEL_NAME}_linked" COMMAND model_linked)
add_test(NAME "${MODEL_NAME}_open" COMMAND model_open "${MODEL_LIBRARY}")
if(WIN32)
  set_tests_properties("${MODEL_NAME}_linked" PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MODEL_DIR}")
endif()
