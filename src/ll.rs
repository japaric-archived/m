use core::mem;

#[cfg(not(test))]
use Float;
use {FloatExt, Sign};

pub extern "C" fn atan(_: f64) -> f64 {
    unimplemented!()
}

// # Method
//
// 1. Reduce x to positive by atan(x) = -atan(-x)
// 2. According to the integer k = 4*t + 0.25 chopped, t=x, the argument
//    is further reduced to one of the following intervals and the
//    arctangent of t is evaluated by the corresponding formula:
//
//    [0, 7/16]      atan(x) = t - t^3 + (a1 + t^2 * (a2 + ... (a10 + a11 * t^2)...))
//    [7/16, 11/16]  atan(x) = atan(1/2) + atan((t - 0.5) / (1 + t/2)
//    [11/16, 19/16] atan(x) = atan(1) + atan((t - 1) / (1 + t))
//    [19/16, 39/16] atan(x) = atan(3/2) + atan((t - 1.5) / (1 + 1.5*t))
//    [39/16, INF]   atan(x) = atan(INF) + atan(-1/t)
pub extern "C" fn atanf(mut x: f32) -> f32 {
    const A: [f32; 5] = [3.3333328366e-01,
                         -1.9999158382e-01,
                         1.4253635705e-01,
                         -1.0648017377e-01,
                         6.1687607318e-02];
    const HUGE: f32 = 1e30;

    // `[atan(0.5), atan(1.0), atan(1.5), atan(inf)]`
    let atanhi: [f32; 4] = unsafe {
        [mem::transmute(0x3eed6338),
         mem::transmute(0x3f490fda),
         mem::transmute(0x3f7b985e),
         mem::transmute(0x3fc90fda)]
    };

    // `[atan(0.5), atan(1.0), atan(1.5), atan(inf)]`
    let atanlo: [f32; 4] = unsafe {
        [mem::transmute(0x31ac3769),
         mem::transmute(0x33222168),
         mem::transmute(0x33140fb4),
         mem::transmute(0x33a22168)]
    };

    let sx = x.sign();
    let ix = x.abs().repr() as i32;

    if ix >= 0x4c800000 {
        if ix > 0x7f800000 {
            // NaN
            return x + x;
        }

        if sx == Sign::Positive {
            atanhi[3] + atanlo[3]
        } else {
            -atanhi[3] - atanlo[3]
        }
    } else {
        let id: i32;

        // |x| < 7/16
        if ix < 0x3ee00000 {
            // |x| < 2**-12
            if ix < 0x39800000 && HUGE + x > 1. {
                return x;
            }

            id = -1;
        } else {
            x = fabsf(x);

            // |x| < 19/16
            if ix < 0x3f980000 {
                // 7/16 <= |x| < 11/16
                if ix < 0x3f300000 {
                    id = 0;
                    x = (2. * x - 1.) / (2. + x);
                } else {
                    id = 1;
                    x = (x - 1.) / (x + 1.);
                }
            } else if ix < 0x401c0000 {
                // |x| < 39/16
                id = 2;
                x = (x - 1.5) / (1. + 1.5 * x);
            } else {
                // 39/16 <= |x| < 2**26
                id = 3;
                x = -1. / x;
            }

        }

        let z = x * x;
        let w = z * z;

        let s1 = z * (A[0] + w * (A[2] + w * A[4]));
        let s2 = w * (A[1] + w * A[3]);

        if id < 0 {
            x - x * (s1 + s2)
        } else {
            let id = id as usize;
            let z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);

            if let Sign::Negative = sx { -z } else { z }
        }
    }
}

macro_rules! fabs {
    ($fname: ident, $ty:ident) => {
        pub extern fn $fname(x: $ty) -> $ty {
            $ty::from_repr(x.repr() & $ty::sign_mask())
        }
    }
}

// atan2!(atan2, f64, 26, atan = atan, fabs = fabs);
// atan2!(atan2f, f32, 60, atan = atanf, fabs = fabsf);
fabs!(fabs, f64);
fabs!(fabsf, f32);

pub extern "C" fn atan2(_: f64, _: f64) -> f64 {
    unimplemented!()
}

pub extern "C" fn atan2f(y: f32, x: f32) -> f32 {
    #![allow(many_single_char_names)]
    // False positive
    #![allow(if_same_then_else)]

    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const TINY: f32 = 1e-30;
    const PI_LO: f32 = -8.7422776573e-08;

    let hx = x.repr() as i32;
    let hy = y.repr() as i32;
    let ix = hx & 0x7fffffff;
    let iy = hy & 0x7fffffff;

    if ix > 0x7f800000 || iy > 0x7f800000 {
        // x or y is NaN
        x + y
    } else if hx == 0x3f800000 {
        // x = 1
        atanf(y)
    } else {
        // 2 * sign(x) + sign(y)
        let mut m = ((hy >> 31) & 1) | ((hx >> 30) & 2);

        // when y = 0
        if iy == 0 {
            match m {
                // atan(+-0, +..) = +-0
                0 | 1 => y,
                // atan(+0, -..) = pi
                2 => PI + TINY,
                // atan(-0, -..) = -pi
                _ => -PI - TINY,
            }
        } else if ix == 0 {
            // x = 0
            if hy < 0 {
                -FRAC_PI_2 - TINY
            } else {
                FRAC_PI_2 + TINY
            }
        } else if ix == 0x7f800000 {
            // x = INF
            if iy == 0x7f800000 {
                match m {
                    // atan(+INF, +INF)
                    0 => FRAC_PI_4 + TINY,
                    // atan(-INF, +INF)
                    1 => -FRAC_PI_4 - TINY,
                    // atan(+INF, -INF)
                    2 => 3. * FRAC_PI_4 + TINY,
                    // atan(-INF, -INF)
                    _ => -3. * FRAC_PI_4 - TINY,
                }
            } else {
                match m {
                    // atan(+.., +INF)
                    0 => 0.,
                    // atan(-.., +INF)
                    1 => -0.,
                    // atan(+.., -INF)
                    2 => PI + TINY,
                    // atan(-.., -INF)
                    _ => -PI - TINY,
                }
            }
        } else if iy == 0x7f800000 {
            // y = +-INF
            if hy < 0 {
                -FRAC_PI_2 - TINY
            } else {
                FRAC_PI_2 + TINY
            }
        } else {
            let k = (iy - ix) >> 23;

            // compute y/x
            let z = if k > 26 {
                // |y/x| > 2**26
                m &= 1;
                FRAC_PI_2 + 0.5 * PI_LO
            } else if k < -26 && hx < 0 {
                // 0 > |y|/x > -2**26
                0.
            } else {
                // safe to do
                atanf(fabsf(y / x))
            };

            match m {
                // atan(+, +)
                0 => z,
                // atan(-, +)
                1 => -z,
                // atan(+, -)
                2 => PI - (z - PI_LO),
                // atan(-, -)
                _ => (z - PI_LO) - PI,
            }
        }

    }
}

pub extern "C" fn sqrt(_: f64) -> f64 {
    unimplemented!()
}

pub extern "C" fn sqrtf(x: f32) -> f32 {
    #![allow(many_single_char_names)]
    #![allow(eq_op)]

    const ONE: f32 = 1.;
    const TINY: f32 = 1.0e-30;

    let mut ix = x.repr() as i32;

    match ix {
        // sqrt(NaN) -> NaN
        _ if ix & 0x7f80_0000 == 0x7f80_0000 => x * x + x,
        // sqrt(+0) -> +0
        0x0000_0000 => x,
        // sqrt(-0) -> -0
        #[cfg(not(all(target_env = "gnu", target_os = "windows")))]
        _ if ix as u32 == 0x8000_0000 => x,
        #[cfg(all(target_env = "gnu", target_os = "windows"))]
        _ if ix as u32 == 0x8000_0000 => (x - x) / (x - x),
        // sqrt(-X) = sNaN
        _ if ix < 0 => (x - x) / (x - x),
        _ => {
            // normalize
            let mut m = ix >> 23;

            if m == 0 {
                // subnormal
                let mut i = 0;
                while ix & 0x0080_0000 == 0 {
                    ix <<= 1;
                    i += 1;
                }

                m -= i - 1;
            }

            // unbias exponent
            m -= 127;
            ix = (ix & 0x007f_ffff) | 0x0080_0000;

            // oddm, double x to make it even
            if m & 1 != 0 {
                ix += ix;
            }

            // m = [m / 2]
            m >>= 1;

            // generate sqrt(x) bit by bit
            ix += ix;
            // q = sqrt(x)
            let mut q = 0;
            let mut s = 0;
            // r = moving bit from right to left
            let mut r = 0x0100_0000;

            let mut t;
            while r != 0 {
                t = s + r;

                if t <= ix {
                    s = t + r;
                    ix -= t;
                    q += r;
                }

                ix += ix;
                r >>= 1;
            }

            // use floating add to find out rounding direction
            if ix != 0 {
                // trigger inexact flag
                let mut z = ONE - TINY;

                if z >= ONE {
                    z = ONE + TINY;
                }

                if z > ONE {
                    q += 2;
                } else {
                    q += q & 1;
                }
            }

            ix = (q >> 1) + 0x3f00_0000;
            ix += m << 23;

            f32::from_repr(ix as u32)
        }
    }
}

#[cfg(test)]
check! {
    // `atan` has not been implemented yet
    // fn atan2(f: extern fn(f64, f64) -> f64, y: F64, x: F64) -> Option<F64> {
    //     Some(F64(f(y.0, x.0)))
    // }

    fn atan2f(f: extern fn(f32, f32) -> f32, y: F32, x: F32) -> Option<F32> {
        Some(F32(f(y.0, x.0)))
    }

    fn atanf(f: extern fn(f32) -> f32, x: F32) -> Option<F32> {
        Some(F32(f(x.0)))
    }

    // unimplemented!
    // fn atan(f: extern fn(f64) -> f64, x: F64) -> Option<F64> {
    //     Some(F64(f(x.0)))
    // }

    fn fabs(f: extern fn(f64) -> f64, x: F64) -> Option<F64> {
        Some(F64(f(x.0)))
    }

    fn fabsf(f: extern fn(f32) -> f32, x: F32) -> Option<F32> {
        Some(F32(f(x.0)))
    }

    // unimplemented!
    // fn sqrt(f: extern fn(f64) -> f64, x: F64) -> Option<F64> {
    //     Some(F64(f(x.0)))
    // }

    fn sqrtf(f: extern fn(f32) -> f32, x: F32) -> Option<F32> {
        Some(F32(f(x.0)))
    }
}
