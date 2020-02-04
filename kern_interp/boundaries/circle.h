// Copyright 2019 John Paul Ryan
#ifndef KERN_INTERP_BOUNDARIES_CIRCLE_H_
#define KERN_INTERP_BOUNDARIES_CIRCLE_H_

#include <memory>
#include "kern_interp/boundaries/boundary.h"

namespace kern_interp {

class Circle : public Boundary {
 public:
  void initialize(int N, BoundaryCondition bc) override;
  bool is_in_domain(const Vec2& a) const override;
  std::unique_ptr<Boundary> clone() const override {
    return std::make_unique<Circle>(*this);
  }
};

}  // namespace kern_interp

#endif  // KERN_INTERP_BOUNDARIES_CIRCLE_H_
