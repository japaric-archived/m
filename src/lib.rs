//! A C free / pure Rust mathematical library ("libm") for `no_std` code
//!
//! This is a port of [OpenLibm].
//!
//! [OpenLibm]: https://github.com/JuliaLang/openlibm
//!
//! # Usage
//!
//! Currently, this crate only provides a `Float` extension trait that's very similar to the one in
//! std but its method are implemented in pure Rust instead of its methods being wrappers over the
//! system libm. For this reason, this trait is usable in C free, `no_std` environments.
//!
//! ```
//! #![no_std]
//!
//! use m::Float as _0;
//!
//! fn foo(x: f32) -> f32 {
//!     x.atan()
//! }
//! ```
//!
//! Mind you that, at the moment, this extension trait only provides a handful of mathematical
//! functions -- only the ones that have been ported to Rust.
//!
//! # Coverage
//!
//! So far, these functions have been ported to Rust:
//!
//! - `atan2f`
//! - `atanf`
//! - `fabs`
//! - `fabsf`

#![cfg_attr(not(test), no_std)]
#![deny(warnings)]

#[cfg(test)]
extern crate core;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use core::mem;
use core::{f32, f64};

#[cfg(test)]
mod m;

#[cfg(test)]
#[macro_use]
mod qc;

mod ll;

/// Trait that provides mathematical functions for floating point numbers
///
/// # Fine print
///
/// This trait is meant to be a "closed" extension trait that's not meant to be implemented by
/// downstream users. As such this trait is *exempted* from semver rules in the context of *adding*
/// new methods to it. Therefore: if you implement this trait (don't do that), your code may (will!)
/// break during a minor version bump. You have been warned!
pub trait Float {
    /// Computes the absolute value of `self`. Returns `NAN` if the number is `NAN`.
    fn abs(self) -> Self;

    /// Computes the arctangent of a number. Return value is in radians in the range `[-pi/2, pi/2]`
    fn atan(self) -> Self;

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`)
    ///
    /// - `x = 0`, `y = 0`: `0`
    /// - `x >= 0`: `arctan(y / x)` -> `[-pi/2, pi/2]`
    /// - `y >= 0`: `arctan(y / x) + pi` -> `(pi/2, pi]`
    /// - `y < 0`: `arctan(y / x) - pi` -> `(-pi, -pi/2)`
    fn atan2(self, Self) -> Self;

    /// Returns `true` if this value is positive infinity or negative infinity and `false`
    /// otherwise.
    fn is_infinite(self) -> bool;

    /// Returns `true` if this value is `NaN` and `false` otherwise.
    fn is_nan(self) -> bool;
}

macro_rules! float {
    ($ty:ident, atan = $atan:ident, atan2 = $atan2:ident, fabs = $fabs:ident) => {
        impl Float for $ty {
            fn abs(self) -> Self {
                ll::$fabs(self)
            }

            fn atan(self) -> Self {
                ll::$atan(self)
            }

            fn atan2(self, other: Self) -> Self {
                ll::$atan2(self, other)
            }

            fn is_infinite(self) -> bool {
                self == $ty::INFINITY || self == $ty::NEG_INFINITY
            }

            fn is_nan(self) -> bool {
                self != self
            }
        }

    }
}

float!(f32, atan = atanf, atan2 = atan2f, fabs = fabsf);
float!(f64, atan = atan, atan2 = atan2, fabs = fabs);

trait FloatExt {
    type Int;

    fn bits() -> u32;
    #[cfg(test)]
    fn eq_repr(self, Self) -> bool;
    fn exponent(self) -> i16;
    fn exponent_bias() -> u32;
    fn exponent_bits() -> u32;
    fn exponent_mask() -> Self::Int;
    fn from_parts(sign: Sign, exponent: Self::Int, significand: Self::Int) -> Self;
    fn from_repr(Self::Int) -> Self;
    fn repr(self) -> Self::Int;
    fn sign(self) -> Sign;
    fn sign_mask() -> Self::Int;
    fn significand_bits() -> u32;
    fn significand_mask() -> Self::Int;
}

macro_rules! float_ext {
    ($float_ty:ident,
     repr_ty = $repr_ty:ident,
     exponent_bits = $exponent_bits:expr,
     significand_bits = $significand_bits:expr) => {
        impl FloatExt for $float_ty {
            type Int = $repr_ty;

            fn bits() -> u32 {
                1 + Self::exponent_bits() + Self::significand_bits()
            }

            #[cfg(test)]
            fn eq_repr(self, rhs: Self) -> bool {
                if self.is_nan() && rhs.is_nan() {
                    true
                } else {
                    let (lhs, rhs) = (self.repr(), rhs.repr());

                    lhs == rhs || (lhs > rhs && lhs - rhs == 1) || (rhs > lhs && rhs - lhs == 1)
                }
            }

            fn exponent(self) -> i16 {
                ((self.repr() & Self::exponent_mask()) >> Self::significand_bits()) as i16 -
                    Self::exponent_bias() as i16
            }

            fn exponent_bias() -> u32 {
                (1 << (Self::exponent_bits() - 1)) - 1
            }

            fn exponent_bits() -> u32 {
                $exponent_bits
            }

            fn exponent_mask() -> Self::Int {
                ((1 << Self::exponent_bits()) - 1) << Self::significand_bits()
            }

            fn from_parts(sign: Sign, exponent: Self::Int, significand: Self::Int) -> Self {
                Self::from_repr(sign.$repr_ty() |
                                exponent & Self::exponent_mask() |
                                significand & Self::significand_mask())
            }

            fn from_repr(x: Self::Int) -> Self {
                unsafe { mem::transmute(x) }
            }

            fn repr(self) -> Self::Int {
                unsafe { mem::transmute(self) }
            }

            fn sign(self) -> Sign {
                if self.repr() >> (Self::bits() - 1) == 0 {
                    Sign::Positive
                } else {
                    Sign::Negative
                }
            }

            fn sign_mask() -> Self::Int {
                (1 << Self::bits() - 1) - 1
            }

            fn significand_bits() -> u32 {
                $significand_bits
            }

            fn significand_mask() -> Self::Int {
                (1 << Self::significand_bits()) - 1
            }
        }
    }
}

float_ext!(f32, repr_ty = u32, exponent_bits = 8, significand_bits = 23);
float_ext!(f64,
           repr_ty = u64,
           exponent_bits = 11,
           significand_bits = 52);

#[derive(Eq, PartialEq)]
enum Sign {
    Negative,
    Positive,
}

impl Sign {
    #[cfg(test)]
    fn from_bool(x: bool) -> Self {
        if x { Sign::Negative } else { Sign::Positive }
    }

    fn u32(self) -> u32 {
        match self {
            Sign::Positive => 0,
            Sign::Negative => 1 << 31,
        }
    }

    fn u64(self) -> u64 {
        match self {
            Sign::Positive => 0,
            Sign::Negative => 1 << 63,
        }
    }
}
