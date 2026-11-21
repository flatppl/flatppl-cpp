#include "poisson_regression.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using flatppl::poisson_regression::Model;
using flatppl::poisson_regression::Outputs;

using Real = decltype(Outputs::log_posterior);
using Count =
    std::conditional_t<std::is_same_v<Real, float>, std::int32_t, std::int64_t>;

template <std::size_t N>
void check(Model& model, const std::vector<Real>& x, const Real (&beta)[2],
           const Count (&counts)[N]) {
  double expected = -std::log(2.0 * std::acos(-1.0));
  for (const double coefficient : beta) {
    expected -= 0.5 * coefficient * coefficient;
  }
  for (std::size_t row = 0; row < N; ++row) {
    const double eta = static_cast<double>(x[row * 2]) * beta[0] +
                       static_cast<double>(x[row * 2 + 1]) * beta[1];
    const double count = static_cast<double>(counts[row]);
    expected += count * eta - std::exp(eta) - std::lgamma(count + 1.0);
  }
  const double actual =
      model.evaluate(static_cast<Count>(N), x, beta, counts).log_posterior;
  const double tolerance =
      64 * std::numeric_limits<Real>::epsilon() * std::abs(expected);
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error("wrong Poisson regression log posterior");
  }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
#ifdef FLATPPL_TEST_LINKED
  auto model = Model::linked();
#else
  if (argc != 2) return 2;
  auto model = Model::open(argv[1]);
#endif
  const Real beta[] = {0.25, 0.5};
  const Count counts[] = {0, 2, 4};
  check(model, {1, -1, 1, 0, 1, 2}, beta, counts);
  const Count zero[] = {0};
  check(model, {1, -1}, beta, zero);

  const std::vector<Real> partial_row{1};
  try {
    model.evaluate(1, partial_row, beta, zero);
    throw std::logic_error("accepted a partial matrix row");
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find("x: wrong buffer size") ==
        std::string::npos)
      throw;
  }
}
