// Copyright 2019 John Paul Ryan
#include <string>
#include <cmath>
#include "kern_interp/vec2.h"

namespace kern_interp {

Vec2::Vec2() {
  a[0] = 0;
  a[1] = 0;
}


Vec2::Vec2(double m, double n) {
  a[0] = m;
  a[1] = n;
}


double Vec2::norm() const {
  return sqrt(a[0] * a[0] + a[1] * a[1]);
}


double Vec2::dot(const Vec2& o) const {
  return a[0] * o.a[0] + a[1] * o.a[1];
}


Vec2 Vec2::operator-(const Vec2 &o) const {
  return Vec2(a[0] - o.a[0], a[1] - o.a[1]);
}


Vec2 Vec2::operator*(const double d) const {
  return Vec2(a[0] * d, a[1] * d);
}

}  // namespace kern_interp
