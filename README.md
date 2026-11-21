# FlatPPL C++ CPU prototype

This prototype builds a complete FlatPPL query into a native shared library
and typed C++ header. The library embeds the query and its IREE CPU runtime.
The command-line interface and generated APIs may change.

## Build

Use Python 3.9 or newer, CMake 3.26 or newer, a native C/C++ toolchain, a
FlatPPL executable, and an IREE compiler with matching IREE source. The local
IREE checkout needs its `benchmark`, `flatcc`, and `printf` submodules.

```sh
./flatppl-build examples/normal.flatppl --out-dir build/normal \
  --flatppl /path/to/flatppl \
  --iree-dir /path/to/iree \
  --iree-compile /path/to/iree-compile
```

The command inherits FlatPPL's precision. Set `--dtype f64` to build a double
precision library. Use a separate output directory for each precision. Set
`--name` to give linked models distinct names.

Use the fetch flags to download pinned tools explicitly. The command builds
FlatPPL from source and downloads the IREE compiler. It caches the tools and
shares the CPU runtime build across models.

```sh
./flatppl-build examples/normal.flatppl --out-dir build/normal \
  --fetch-flatppl --fetch-iree
```

Add `--build-iree-from-source` to build the compiler from the selected IREE
source. Use `--iree-dir` with a prepared checkout to build offline. Add
`--fetch-iree` to authorize fetching source and submodules for the source build.

## Call the model

Put the generated header directory on the include path. The header works
with C++17 and later. Add the generated library to the link command when using
`Model::linked()`. Use the platform's dynamic loading library when using
`Model::open()`, such as `${CMAKE_DL_LIBS}` in CMake.

```cpp
#include "normal.hpp"

float score(const char* library_path) {
  auto model = flatppl::normal::Model::open(library_path);
  const float observations[] = {0, 1, 2, 3};
  const auto result = model.evaluate({4, {observations, 4, {4}}, 1.0f});
  return result.total;
}
```

The generated names follow the query signature. Record and table fields
flatten into ordered tensor leaves. Pass tensors in dense row-major order.
The tensor view carries an element count and shape. A dynamic dimension may
vary between calls. The generated wrapper also checks a leading dimension
declared directly by another scalar input, as in the Normal example.

Reuse a model instance across calls. Pass observations and parameters each
time. Give concurrent callers separate instances. C++ results own their data.
The C API in `flatppl.h` keeps results until the next call on that context.

The prototype supports dense real, integer, and Boolean tensor queries that
lower through the existing StableHLO log-density mode. It reports incompatible
signatures and propagates IREE errors and FlatPPL runtime status.

## Verify

Build the Normal example for each precision, then point the test at its
output directory.

```sh
cmake -S tests -B build/tests -DMODEL_DIR="$PWD/build/normal"
cmake --build build/tests --config Release
ctest --test-dir build/tests -C Release --output-on-failure
```

Both consumers compare the same model library against the analytic Normal
log likelihood at two observation counts and check rejection of an inconsistent
count.
