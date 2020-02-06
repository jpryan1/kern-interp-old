// Copyright 2019 John Paul Ryan
#ifndef KERN_INTERP_VEC2_H_
#define KERN_INTERP_VEC2_H_

namespace kern_interp {

struct Vec2 {
  double a[2];

  Vec2();
  Vec2(double m, double n);
  double norm() const;
  double dot(const Vec2& o) const;
  Vec2 operator-(const Vec2 &o) const;
  Vec2 operator*(const double d) const;
};

}  // namespace kern_interp

#endif  // KERN_INTERP_VEC2_H_
