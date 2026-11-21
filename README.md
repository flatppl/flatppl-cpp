# FlatPPL C++ CPU prototype

This prototype builds a complete FlatPPL query into a native shared library
and typed C++ header. The library embeds the query and its IREE CPU runtime.
The command-line interface and generated APIs may change.

## Build

Build the [Poisson regression example](examples/poisson_regression.flatppl)
to evaluate the unnormalized log posterior of an intercept and a slope.
The model adds the log likelihood of the observed counts to the log prior.

Use Python 3.9 or newer, CMake 3.26 or newer, a native C/C++ toolchain, a
FlatPPL executable, and an IREE compiler with matching IREE source. The local
IREE checkout needs its `benchmark`, `flatcc`, and `printf` submodules.

```sh
./flatppl-build examples/poisson_regression.flatppl --out-dir build/poisson_regression \
  --flatppl /path/to/flatppl \
  --iree-dir /path/to/iree \
  --iree-compile /path/to/iree-compile
```

The command inherits FlatPPL's precision. Use the default `f32` for the Poisson
example. Use `--dtype f64` for models supported in double precision. Use a
separate output directory for each precision. Set `--name` to give linked
models distinct names.

Use the fetch flags to download pinned tools explicitly. The command builds
FlatPPL from source and downloads the IREE compiler. It caches the tools and
shares the CPU runtime build across models.

```sh
./flatppl-build examples/poisson_regression.flatppl --out-dir build/poisson_regression \
  --fetch-flatppl --fetch-iree
```

Add `--build-iree-from-source` to build the compiler from the selected IREE
source. Use `--iree-dir` with a prepared checkout to build offline. Add
`--fetch-iree` to authorize fetching source and submodules for the source build.

See [IREE workarounds](docs/workarounds.md) for the StableHLO changes applied
before compilation.

## Call the model

The default `f32` build takes a flat row-major design matrix with columns
`[1, predictor]`. Pass one coefficient per column and one count per row.

```cpp
#include "poisson_regression.hpp"

#include <cstdint>
#include <vector>

float log_posterior(const char* library_path) {
  auto model = flatppl::poisson_regression::Model::open(library_path);
  const std::vector<float> x = {1, -1, 1, 0, 1, 2};
  const float beta[] = {0.25f, 0.5f};
  const std::int32_t counts[] = {0, 2, 4};
  return model.evaluate(3, x, beta, counts).log_posterior;
}
```

The output directory contains a CMake package. Set `poisson_regression_DIR`
to that directory when configuring the caller. Both targets provide the
headers and require C++17.

```cmake
find_package(poisson_regression CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE flatppl::poisson_regression::headers)
```

Use `flatppl::poisson_regression` with `Model::linked()` to link the model
directly. Use the `::headers` target with `Model::open(path)` to load it
at runtime.

Pass arguments in the order declared by the FlatPPL query. The wrapper infers
the shape of a one-dimensional `std::array` or `std::vector`, or a C array
with one or two dimensions. A matrix with dynamic rows and a fixed column
count also accepts a flat `std::vector`. Use an explicit tensor view for other
shapes or raw pointers. Pass one generated argument per record or table with
multiple fields. The wrapper packs its fields into ordered tensor leaves.

Pass tensors in dense row-major order. A dynamic dimension may vary between
calls. The wrapper checks a leading dimension declared directly by another
scalar input, as in the Normal example.

Reuse a model instance across calls. Pass observations and parameters each
time. Give concurrent callers separate instances. C++ results own their data.
The C API in `flatppl.h` keeps results until the next call on that context.

The prototype supports dense real, integer, and Boolean tensor queries that
lower through the existing StableHLO log-density mode. It reports incompatible
signatures and propagates IREE errors and FlatPPL runtime status.

## Verify

Build the Poisson example, then point the tests at its output directory.

```sh
cmake -S tests -B build/tests -DMODEL_DIR="$PWD/build/poisson_regression"
cmake --build build/tests --config Release
ctest --test-dir build/tests -C Release --output-on-failure
```

Both consumers compare the log posterior against the Normal prior and Poisson
likelihood formulas. They also check that the wrapper rejects an incomplete row.

Build `examples/linear_regression.flatppl` and configure the same tests with
`-DMODEL_NAME=linear_regression`. The model takes a row count, an `N×P`
design matrix, a length-`P` coefficient vector, and a length-`N` response
vector. The C++ test calls `model.evaluate(n, x, beta, y)` with 4, 32, 128,
and 1024 rows on the same model.

## License

[MIT](LICENSE)
