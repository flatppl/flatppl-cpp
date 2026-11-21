#include "normal.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>

using flatppl::normal::Inputs;
using flatppl::normal::Model;

using Real = decltype(Inputs::mu);
using Count = decltype(Inputs::n);

template <std::size_t N>
void check(Model& model, const std::array<Real, N>& observations, Real mu) {
  const double log_normalizer = 0.5 * std::log(2.0 * std::acos(-1.0));
  double expected = 0;
  for (Real observation : observations) {
    const double residual = static_cast<double>(observation) - mu;
    expected -= log_normalizer + 0.5 * residual * residual;
  }
  Inputs inputs{static_cast<Count>(N), {observations.data(), N, {N}}, mu};
  const double actual = model.evaluate(inputs).total;
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
  const std::array<Real, 1> single{0};
  const std::array<Real, 4> multiple{0, 1, 2, 3};
  check(model, single, Real{0});
  check(model, multiple, Real{1});

  const Inputs wrong_count{
      1, {multiple.data(), multiple.size(), {multiple.size()}}, Real{1}};
  try {
    model.evaluate(wrong_count);
    throw std::logic_error("accepted an inconsistent observation count");
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find("dimension disagrees") ==
        std::string::npos)
      throw;
  }
  check(model, multiple, Real{2});
  std::puts("Normal query passed at lengths 1 and 4");
}
