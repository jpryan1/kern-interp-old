// Copyright 2019 John Paul Ryan
#ifndef KERN_INTERP_BOUNDARIES_DONUT_H_
#define KERN_INTERP_BOUNDARIES_DONUT_H_

#include <memory>
#include "kern_interp/boundaries/boundary.h"

namespace kern_interp {

class Donut : public Boundary {
 public:
  void initialize(int N, BoundaryCondition bc) override;
  bool is_in_domain(const Vec2& a) const override;
  std::unique_ptr<Boundary> clone() const override {
    return std::make_unique<Donut>(*this);
  }
};

}  // namespace kern_interp

#endif  // KERN_INTERP_BOUNDARIES_DONUT_H_
