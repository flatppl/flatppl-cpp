cmake_minimum_required(VERSION 3.26)
project(flatppl_cpp_tests LANGUAGES CXX)
enable_testing()

find_file(MODEL_LIBRARY
  NAMES "${CMAKE_SHARED_LIBRARY_PREFIX}normal${CMAKE_SHARED_LIBRARY_SUFFIX}"
  PATHS "${MODEL_DIR}" NO_DEFAULT_PATH REQUIRED NO_CACHE)

add_library(normal_model SHARED IMPORTED)
set_target_properties(normal_model PROPERTIES IMPORTED_LOCATION "${MODEL_LIBRARY}")
if(WIN32)
  find_library(MODEL_IMPLIB NAMES normal PATHS "${MODEL_DIR}"
    NO_DEFAULT_PATH REQUIRED NO_CACHE)
  set_target_properties(normal_model PROPERTIES IMPORTED_IMPLIB "${MODEL_IMPLIB}")
endif()

foreach(mode linked open)
  add_executable("normal_${mode}" normal.cc)
  target_include_directories("normal_${mode}" PRIVATE "${MODEL_DIR}")
  set_target_properties("normal_${mode}" PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options("normal_${mode}" PRIVATE -Wall -Wextra -Werror)
  endif()
endforeach()

target_compile_definitions(normal_linked PRIVATE FLATPPL_TEST_LINKED)
target_link_libraries(normal_linked PRIVATE normal_model)
target_link_libraries(normal_open PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME normal_linked COMMAND normal_linked)
add_test(NAME normal_open COMMAND normal_open "${MODEL_LIBRARY}")
if(WIN32)
  set_tests_properties(normal_linked PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MODEL_DIR}")
endif()
