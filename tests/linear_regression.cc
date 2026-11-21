#include "linear_regression.hpp"

#include <cmath>
#include <limits>
#include <string>

using flatppl::linear_regression::Model;
using flatppl::linear_regression::Outputs;

using Real = decltype(Outputs::total);

template <std::size_t N>
void check(Model& model) {
  Real x[N][2];
  Real y[N];
  const Real beta[] = {1.5, -0.5};
  std::vector<Real> predicted;
  predicted.reserve(N);
  const double log_normalizer = 0.5 * std::log(2.0 * std::acos(-1.0));
  double expected_total = 0;
  for (std::size_t row = 0; row < N; ++row) {
    x[row][0] = static_cast<int>(row % 13) - 6;
    x[row][1] = static_cast<int>(row % 7) - 3;
    y[row] = 0.25 * static_cast<double>(row % 17);
    const double mean = x[row][0] * 1.5 - x[row][1] * 0.5;
    const double residual = y[row] - mean;
    predicted.push_back(static_cast<Real>(mean));
    expected_total -= log_normalizer + 0.5 * residual * residual;
  }

  const auto result = model.evaluate(static_cast<std::int32_t>(N), x, beta, y);
  if (result.prediction.shape != std::array<std::uint64_t, 1>{N} ||
      result.prediction.data != predicted) {
    throw std::runtime_error("wrong regression prediction");
  }
  // Allow sequential summation of N same-sign terms in the emitted type.
  const double roundoff = (N + 4) * std::numeric_limits<Real>::epsilon() / 2;
  const double tolerance = roundoff / (1 - roundoff) * std::abs(expected_total);
  if (!std::isfinite(result.total) ||
      std::abs(result.total - expected_total) > tolerance) {
    throw std::runtime_error("wrong regression log likelihood");
  }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
#ifdef FLATPPL_TEST_LINKED
  auto model = Model::linked();
#else
  if (argc != 2) return 2;
  auto model = Model::open(argv[1]);
#endif
  check<4>(model);
  check<32>(model);
  check<128>(model);
  check<1024>(model);

  const Real wrong_rows[3][2] = {};
  const Real beta[2] = {};
  const Real y[4] = {};
  try {
    model.evaluate(4, wrong_rows, beta, y);
    throw std::logic_error("accepted an inconsistent matrix row count");
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find("x: dimension disagrees with n") ==
        std::string::npos)
      throw;
  }
}
