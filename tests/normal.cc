#include "normal.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <string>
#include <type_traits>

using flatppl::normal::Model;
using flatppl::normal::Outputs;

using Real = decltype(Outputs::total);

template <typename Container>
void check(Model& model, const Container& observations, Real mu) {
  const double log_normalizer = 0.5 * std::log(2.0 * std::acos(-1.0));
  double expected = 0;
  for (Real observation : observations) {
    const double residual = static_cast<double>(observation) - mu;
    expected -= log_normalizer + 0.5 * residual * residual;
  }
  const auto count = static_cast<std::int32_t>(std::size(observations));
  const double actual = model.evaluate(count, observations, mu).total;
  const double tolerance = std::is_same<Real, float>::value ? 1e-6 : 1e-13;
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(
        "Normal log likelihood disagrees with the analytic result");
  }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
#ifdef FLATPPL_TEST_LINKED
  auto model = Model::linked();
#else
  if (argc != 2) return 2;
  auto model = Model::open(argv[1]);
#endif
  const Real single[] = {0};
  const std::array<Real, 4> multiple{0, 1, 2, 3};
  check(model, single, Real{0});
  check(model, multiple, Real{1});

  try {
    model.evaluate(1, multiple, Real{1});
    throw std::logic_error("accepted an inconsistent observation count");
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find("dimension disagrees") ==
        std::string::npos)
      throw;
  }
  check(model, std::vector<Real>(multiple.begin(), multiple.end()), Real{2});
  std::puts("Normal query passed at lengths 1 and 4");
}
