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
    const A: [f32; 5] = [
        3.3333328366e-01,
        -1.9999158382e-01,
        1.4253635705e-01,
        -1.0648017377e-01,
        6.1687607318e-02,
    ];
    const HUGE: f32 = 1e30;

    // `[atan(0.5), atan(1.0), atan(1.5), atan(inf)]`
    let atanhi: [f32; 4] = unsafe {
        [
            mem::transmute(0x3eed6338),
            mem::transmute(0x3f490fda),
            mem::transmute(0x3f7b985e),
            mem::transmute(0x3fc90fda),
        ]
    };

    // `[atan(0.5), atan(1.0), atan(1.5), atan(inf)]`
    let atanlo: [f32; 4] = unsafe {
        [
            mem::transmute(0x31ac3769),
            mem::transmute(0x33222168),
            mem::transmute(0x33140fb4),
            mem::transmute(0x33a22168),
        ]
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

            if let Sign::Negative = sx {
                -z
            } else {
                z
            }
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

    if ix & 0x7f80_0000 == 0x7f80_0000 {
        x * x + x
    } else if ix == 0x0000_0000 || ix as u32 == 0x8000_0000 {
        x
    } else if x < 0. {
        (x - x) / (x - x)
    } else {
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

pub extern "C" fn sin(_: f64) -> f64 {
    unimplemented!()
}

pub extern "C" fn sinf(x: f32) -> f32 {
    use core::f64::consts::FRAC_PI_2;

    const S1PIO2: f64 = 1f64 * FRAC_PI_2; /* 0x3FF921FB, 0x54442D18 */
    const S2PIO2: f64 = 2f64 * FRAC_PI_2; /* 0x400921FB, 0x54442D18 */
    const S3PIO2: f64 = 3f64 * FRAC_PI_2; /* 0x4012D97C, 0x7F3321D2 */
    const S4PIO2: f64 = 4f64 * FRAC_PI_2; /* 0x401921FB, 0x54442D18 */

    let hx = x.repr() as i32;

    let ix = hx & 0x7fffffff;

    if ix <= 0x3f490fda {
        /* |x| ~<= pi/4 */
        if ix < 0x39800000 {
            /* |x| < 2**-12 */
            if (x as i32) == 0 {
                return x; /* x with inexact if x != 0 */
            }
        }
        return sindf(x as f64);
    }
    if ix <= 0x407b53d1 {
        /* |x| ~<= 5*pi/4 */
        if ix <= 0x4016cbe3 {
            /* |x| ~<= 3pi/4 */
            if hx > 0 {
                return cosdf(x as f64 - S1PIO2);
            } else {
                return -cosdf(x as f64 + S1PIO2);
            }
        } else {
            return sindf((if hx > 0 { S2PIO2 } else { -S2PIO2 }) - x as f64);
        }
    }
    if ix <= 0x40e231d5 {
        /* |x| ~<= 9*pi/4 */
        if ix <= 0x40afeddf {
            /* |x| ~<= 7*pi/4 */
            if hx > 0 {
                return -cosdf(x as f64 - S3PIO2);
            } else {
                return cosdf(x as f64 + S3PIO2);
            }
        } else {
            return sindf(x as f64 + (if hx > 0 { -S4PIO2 } else { S4PIO2 }));
        }
    }
        /* sin(Inf or NaN) is NaN */ else if ix >= 0x7f800000 {
        return x - x;


        /* general argument reduction needed */
    } else {
        let (n, y) = ieee754_rem_pio2f(x);
        match n & 3 {
            0 => return sindf(y),
            1 => return cosdf(-y),
            2 => return sindf(-y),
            _ => return -cosdf(y),
        }
    }
}

pub extern "C" fn cos(_: f64) -> f64 {
    unimplemented!()
}

pub extern "C" fn cosf(x: f32) -> f32 {
    use core::f64::consts::FRAC_PI_2;

    const C1PIO2: f64 = 1f64 * FRAC_PI_2; /* 0x3FF921FB, 0x54442D18 */
    const C2PIO2: f64 = 2f64 * FRAC_PI_2; /* 0x400921FB, 0x54442D18 */
    const C3PIO2: f64 = 3f64 * FRAC_PI_2; /* 0x4012D97C, 0x7F3321D2 */
    const C4PIO2: f64 = 4f64 * FRAC_PI_2; /* 0x401921FB, 0x54442D18 */

    let hx = x.repr() as i32;
    let ix = hx & 0x7fffffff;

    if ix <= 0x3f490fda {
        /* |x| ~<= pi/4 */
        if ix < 0x39800000 {
            /* |x| < 2**-12 */
            if x as i32 == 0 {
                return 1.0;
            } /* 1 with inexact if x != 0 */
        }
        return cosdf(x as f64);
    }
    if ix <= 0x407b53d1 {
        /* |x| ~<= 5*pi/4 */
        if ix <= 0x4016cbe3 {
            /* |x|  ~> 3*pi/4 */
            if hx > 0 {
                return sindf(C1PIO2 - x as f64);
            } else {
                return sindf(x as f64 + C1PIO2);
            }
        } else {
            return -cosdf(x as f64 + (if hx > 0 { -C2PIO2 } else { C2PIO2 }));
        }
    }
    if ix <= 0x40e231d5 {
        /* |x| ~<= 9*pi/4 */
        if ix <= 0x40afeddf {
            /* |x|  ~> 7*pi/4 */
            if hx > 0 {
                return sindf(x as f64 - C3PIO2);
            } else {
                return sindf(-C3PIO2 - x as f64);
            }
        } else {
            return cosdf(x as f64 + (if hx > 0 { -C4PIO2 } else { C4PIO2 }));
        }
    }
        /* cos(Inf or NaN) is NaN */ else if ix >= 0x7f800000 {
        return x - x;

        /* general argument reduction needed */
    } else {
        let (n, y) = ieee754_rem_pio2f(x);
        match n & 3 {
            0 => return cosdf(y),
            1 => return sindf(-y),
            2 => return -cosdf(y),
            _ => return sindf(y),
        }
    }
}

fn cosdf(x: f64) -> f32 {
    const ONE: f64 = 1.0;
    const C0: f64 = -0.499999997251031003120;
    const C1: f64 = 0.0416666233237390631894;
    const C2: f64 = -0.00138867637746099294692;
    const C3: f64 = 0.0000243904487962774090654;

    let r: f64;
    let w: f64;
    let z: f64;

    /* Try to optimize for parallel evaluation as in k_tanf.c. */
    z = x * x;
    w = z * z;
    r = C2 + z * C3;
    (((ONE + z * C0) + w * C1) + (w * z) * r) as f32
}

fn sindf(x: f64) -> f32 {
    const S1: f64 = -0.166666666416265235595;
    const S2: f64 = 0.0083333293858894631756;
    const S3: f64 = -0.000198393348360966317347;
    const S4: f64 = 0.0000027183114939898219064;

    let r: f64;
    let s: f64;
    let w: f64;
    let z: f64;

    z = x * x;
    w = z * z;
    r = S3 + z * S4;
    s = z * x;
    ((x + s * (S1 + z * S2)) + s * w * r) as f32
}

fn ieee754_rem_pio2f(x: f32) -> (i32, f64) {
    const HEX18P52: f64 = 6755399441055744.0;
    const INVPIO2: f64 = 6.36619772367581382433e-01; /* 0x3FE45F30, 0x6DC9C883 */
    const PIO2_1: f64 = 1.57079631090164184570e+00; /* 0x3FF921FB, 0x50000000 */
    const PIO2_1T: f64 = 1.58932547735281966916e-08; /* 0x3E5110b4, 0x611A6263 */

    let y: f64;
    let w: f64;
    let r: f64;
    let mut func: f64;
    let z: f32;
    let e0: i32;

    let hx = x.repr() as i32;
    let ix = hx & 0x7fffffff;
    /* 33+53 bit pi is good enough for medium size */
    if ix < 0x4dc90fdb {
        /* |x| ~< 2^28*(pi/2), medium size */
        /* Use a specialized rint() to get func.  Assume round-to-nearest. */
        func = x as f64 * INVPIO2 + HEX18P52;
        func = func - HEX18P52;
        let n = func as i32;
        r = x as f64 - func * PIO2_1;
        w = func * PIO2_1T;
        y = r - w;
        return (n, y);
    }

    /*
     * all other (large) arguments
     */
    if ix >= 0x7f800000 {
        /* x is inf or NaN */
        y = x as f64 - x as f64;
        return (0, y);
    }
    /* set z = scalbn(|x|,ilogb(|x|)-23) */
    e0 = (ix >> 23) - 150; /* e0 = ilogb(|x|)-23; */
    z = f32::from_repr((ix - (e0 << 23)) as u32);
    let (n, ty) = kernel_rem_pio2([z as f64], e0, 1, 0);
    if hx < 0 {
        y = -ty[0];
        return (-n, y);
    }
    y = ty[0];
    (n, y)
}

fn kernel_rem_pio2(x: [f64; 1], e0: i32, nx: i32, prec: i32) -> (i32, [f64; 3]) {
    const IPIO2: [i32; 66] = [
        0xA2F983,
        0x6E4E44,
        0x1529FC,
        0x2757D1,
        0xF534DD,
        0xC0DB62,
        0x95993C,
        0x439041,
        0xFE5163,
        0xABDEBB,
        0xC561B7,
        0x246E3A,
        0x424DD2,
        0xE00649,
        0x2EEA09,
        0xD1921C,
        0xFE1DEB,
        0x1CB129,
        0xA73EE8,
        0x8235F5,
        0x2EBB44,
        0x84E99C,
        0x7026B4,
        0x5F7E41,
        0x3991D6,
        0x398353,
        0x39F49C,
        0x845F8B,
        0xBDF928,
        0x3B1FF8,
        0x97FFDE,
        0x05980F,
        0xEF2F11,
        0x8B5A0A,
        0x6D1F6D,
        0x367ECF,
        0x27CB09,
        0xB74F46,
        0x3F669E,
        0x5FEA2D,
        0x7527BA,
        0xC7EBE5,
        0xF17B3D,
        0x0739F7,
        0x8A5292,
        0xEA6BFB,
        0x5FB11F,
        0x8D5D08,
        0x560330,
        0x46FC7B,
        0x6BABF0,
        0xCFBC20,
        0x9AF436,
        0x1DA9E3,
        0x91615E,
        0xE61B08,
        0x659985,
        0x5F14A0,
        0x68408D,
        0xFFD880,
        0x4D7327,
        0x310606,
        0x1556CA,
        0x73A8C9,
        0x60E27B,
        0xC08C6B,
    ];
    const PIO2: [f64; 8] = [
        1.57079625129699707031e+00, /* 0x3FF921FB, 0x40000000 */
        7.54978941586159635335e-08, /* 0x3E74442D, 0x00000000 */
        5.39030252995776476554e-15, /* 0x3CF84698, 0x80000000 */
        3.28200341580791294123e-22, /* 0x3B78CC51, 0x60000000 */
        1.27065575308067607349e-29, /* 0x39F01B83, 0x80000000 */
        1.22933308981111328932e-36, /* 0x387A2520, 0x40000000 */
        2.73370053816464559624e-44, /* 0x36E38222, 0x80000000 */
        2.16741683877804819444e-51, /* 0x3569F31D, 0x00000000 */
    ];
    const TWO24: f64 = 1.67772160000000000000e+07;
    const TWON24: f64 = 5.96046447753906250000e-08; /* 0x3E700000, 0x00000000 */
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const JK_INIT: [i32; 4] = [3, 4, 4, 6];
    let mut iq: [i32; 20] = [0; 20];
    let mut f: [f64; 20] = [0.0; 20];
    let mut fq: [f64; 20] = [0.0; 20];
    let mut q: [f64; 20] = [0.0; 20];
    let mut y: [f64; 3] = [0.0; 3];
    let mut k: i32;
    let mut z: f64 = 0.0;
    let mut ih: i32 = 0;
    let mut n: i32 = 0;
    let mut carry: i32;

    /* initialize jk*/
    let jk = JK_INIT[prec as usize];
    let jp = JK_INIT[prec as usize];

    /* determine jx,jv,q0, note that 3>q0 */
    let jx: i32 = nx - 1;
    let mut jv: i32 = (e0 - 3) / 24;
    if jv < 0 {
        jv = 0;
    }
    let mut q0: i32 = e0 - 24 * (jv + 1);

    /* set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk] */
    let mut j = jv - jx;
    let m = jx + jk;
    let mut i: i32 = 0;
    while i <= m {
        f[i as usize] = if j < 0 {
            ZERO
        } else {
            IPIO2[j as usize] as f64
        };
        i += 1;
        j += 1;
    }

    i = 0;
    let mut fw: f64;
    /* compute q[0],q[1],...q[jk] */
    while i <= jk {
        j = 0;
        fw = 0.0;
        while j <= jx {
            fw += x[j as usize] * f[(jx + i - j) as usize];
            j += 1;
        }
        q[i as usize] = fw;
        i += 1;
    }

    let mut jz = jk;
    let mut _done: bool = false;
    'recompute: while !_done {
        _done = true;
        /* distill q[] into iq[] reversingly */
        //let mut i: usize = 0;
        i = 0;
        j = jz;
        z = q[jz as usize];
        while j > 0 {
            fw = ((TWON24 * z) as i32) as f64;
            iq[i as usize] = (z - TWO24 * fw) as i32;
            z = q[(j - 1) as usize] + fw;
            i += 1;
            j -= 1;
        }

        /* compute n */
        z = scalbn(z, q0); /* actual value of z */
        z -= 8.0 * floor(z * 0.125_f64); /* trim off integer >= 8 */
        n = z as i32;
        z -= n as f64;
        ih = 0;
        if q0 > 0 {
            /* need iq[jz-1] to determine n */
            i = (iq[jz as usize - 1] >> (24 - q0)) as i32;
            n += i;
            iq[jz as usize - 1] -= (i << (24 - q0)) as i32;
            ih = iq[jz as usize - 1] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[jz as usize - 1] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }

        if ih > 0 {
            /* q > 0.5 */
            n += 1;
            carry = 0;
            i = 0;
            while i < jz {
                j = iq[i as usize];
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        iq[i as usize] = 0x1000000 - j;
                    }
                } else {
                    iq[i as usize] = 0xffffff - j;
                }
                i += 1;
            }

            if q0 > 0 {
                /* rare case: chance is 1 in 12 */
                match q0 {
                    1 => iq[jz as usize - 1] &= 0x7fffff,
                    2 => iq[jz as usize - 1] &= 0x3fffff,
                    _ => iq[jz as usize - 1] &= 0x3fffff,
                }
            }
            if ih == 2 {
                z = ONE - z;
                if carry != 0 {
                    z -= scalbn(ONE, q0);
                }
            }
        }

        /* check if recomputation is needed */
        if z == ZERO {
            j = 0;
            i = jz - 1;
            while i >= jk {
                j |= iq[i as usize];
                i -= 1;
            }
            if j == 0 {
                /* need recomputation */
                k = 1;
                while iq[(jk - k) as usize] == 0 {
                    k += 1;
                } /* k = no. of terms needed */

                i = jz + 1;
                while i <= jz + k {
                    /* add q[jz+1] to q[jz+k] */
                    f[(jx + i) as usize] = IPIO2[(jv + i) as usize] as f64;

                    j = 0;
                    fw = 0.0;
                    while j <= jx {
                        fw += x[j as usize] * f[(jx + i - j) as usize];
                        j += 1;
                    }
                    q[i as usize] = fw;
                    i += 1;
                }
                jz += k;
                _done = false;
                continue 'recompute;
            }
        }
    }

    /* chop off zero terms */
    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq[jz as usize] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        /* break z into 24-bit if necessary */
        z = scalbn(z, -(q0));
        if z >= TWO24 {
            fw = ((TWON24 * z) as i32) as f64;
            iq[jz as usize] = (z - TWO24 * fw) as i32;
            jz += 1;
            q0 += 24;
            iq[jz as usize] = fw as i32;
        } else {
            iq[jz as usize] = z as i32;
        }
    }

    /* convert integer "bit" chunk to floating-point value */
    fw = scalbn(ONE, q0);
    i = jz;
    while i >= 0 {
        q[i as usize] = fw * (iq[i as usize] as f64);
        fw *= TWON24;
        i -= 1;
    }

    /* compute PIo2[0,...,jp]*q[jz,...,0] */
    i = jz;
    while i >= 0 {
        fw = 0.0;
        k = 0;
        while k <= jp && k <= jz - i {
            fw += PIO2[k as usize] * q[(i + k) as usize];
            k += 1;
        }
        fq[(jz - i) as usize] = fw;
        i -= 1;
    }

    /* compress fq[] into y[] */
    match prec {
        0 => {
            fw = 0.0;
            i = jz;
            while i >= 0 {
                fw += fq[i as usize];
                i -= 1;
            }
            y[0] = if ih == 0 { fw } else { -fw };
        }

        1 | 2 => {
            fw = 0.0;
            i = jz;
            while i >= 0 {
                fw += fq[i as usize];
                i -= 1;
            }
            //    STRICT_ASSIGN(double, fw, fw);
            y[0] = if ih == 0 { fw } else { -fw };
            fw = fq[0] - fw;
            fq[0] - fw;
            i = 1;
            while i <= jz {
                fw += fq[i as usize];
                i += 1;
            }
            y[1] = if ih == 0 { fw } else { -fw };
        }

        3 => {
            /* painful */
            i = jz;
            while i > 0 {
                fw = fq[i as usize - 1] + fq[i as usize];
                fq[i as usize] += fq[i as usize - 1] - fw;
                fq[i as usize - 1] = fw;
                i -= 1;
            }
            i = jz;
            while i > 1 {
                fw = fq[i as usize - 1] + fq[i as usize];
                fq[i as usize] += fq[i as usize - 1] - fw;
                fq[i as usize - 1] = fw;
                i -= 1;
            }
            fw = 0.0;
            i = jz;
            while i >= 2 {
                fw += fq[i as usize];
                i -= 1;
            }
            if ih == 0 {
                y[0] = fq[0];
                y[1] = fq[1];
                y[2] = fw;
            } else {
                y[0] = -fq[0];
                y[1] = -fq[1];
                y[2] = -fw;
            }
        }
        _ => {}
    }
    (n & 7, y)
}

pub extern "C" fn floor(x: f64) -> f64 {
    #![allow(overflowing_literals)]
    const HUGE: f64 = 1.0e300;
    let mut i0 = (x.repr() >> 32) as i32;
    let mut i1 = x.repr() as u32;
    let i: u32;
    let j: u32;

    let j0: i32 = ((i0 >> 20) & 0x7ff) - 0x3ff;
    if j0 < 20 {
        if j0 < 0 {
            /* raise inexact if x != 0 */
            if HUGE + x > 0.0 {
                /* return 0*sign(x) if |x|<1 */
                if i0 >= 0 {
                    i0 = 0;
                    i1 = 0;
                } else if (((i0 & 0x7fffffff) as u32) | i1) != 0 {
                    i0 = 0xbff00000;
                    i1 = 0;
                }
            }
        } else {
            i = 0x000fffff >> j0;
            if (((i0 & i as i32) as u32) | i1) == 0 {
                return x;
            } /* x is integral */
            if HUGE + x > 0.0 {
                /* raise inexact flag */
                if i0 < 0 {
                    i0 += (0x00100000) >> j0;
                }
                i0 &= !(i as i32);
                i1 = 0;
            }
        }
    } else if j0 > 51 {
        if j0 == 0x400 {
            return x + x;
        }
            /* inf or NaN */ else {
            return x;
        } /* x is integral */
    } else {
        i = 0xffffffff as u32 >> (j0 - 20);
        if (i1 & i) == 0 {
            return x;
        } /* x is integral */
        if HUGE + x > 0.0 {
            /* raise inexact flag */
            if i0 < 0 {
                if j0 == 20 {
                    i0 += 1;
                } else {
                    let i1x = i1 as u32;
                    j = i1x.wrapping_add(1u32 << (52 - j0 as u32)) as u32;
                    if j < i1 {
                        i0 += 1;
                    } /* got a carry */
                    i1 = j;
                }
            }
            i1 &= !i;
        }
    }

    f64::from_repr(((i0 as u64) << 32) | (i1 as u64))
}

pub extern "C" fn floorf(x: f32) -> f32 {
    #![allow(overflowing_literals)]
    const HUGE: f32 = 1.0e30;
    let mut i0 = x.repr() as i32;
    //let j0: u32;
    let i: i32;


    let j0: i32 = ((i0 >> 23) & 0xff) as i32 - 0x7f_i32;


    if j0 < 23 {
        if j0 < 0 {
            /* raise inexact if x != 0 */
            if HUGE + x > 0.0f32 {
                /* return 0*sign(x) if |x|<1 */
                if i0 >= 0 {
                    i0 = 0;
                } else if (i0 & 0x7fffffff) != 0 {
                    i0 = 0xbf800000;
                }
            }
        } else {
            i = (0x007fffff) >> j0;


            if (i0 & i) == 0 {
                return x; /* x is integral */
            }

            if (HUGE + x) > 0.0_f32 {
                /* raise inexact flag */
                if i0 < 0 {
                    i0 += (0x00800000) >> j0;
                }
                i0 &= !i;
            }
        }
    } else {
        if j0 == 0x80 {
            return x + x; /* inf or NaN */
        } else {
            return x;
        } /* x is integral */
    }

    f32::from_repr(i0 as u32)
}

pub extern "C" fn scalbnf(_: f32, _: i32) -> f32 {
    unimplemented!()
}

pub extern "C" fn scalbn(mut x: f64, n: i32) -> f64 {
    const TWO54: f64 = 1.80143985094819840000e+16; /* 0x43500000, 0x00000000 */
    const TWOM54: f64 = 5.55111512312578270212e-17; /* 0x3C900000, 0x00000000 */
    const HUGE: f64 = 1.0e+300;
    const TINY: f64 = 1.0e-300;

    let mut hx = (x.repr() >> 32) as i32;
    let lx = x.repr() as i32;
    let mut k: i32 = ((hx & 0x7ff00000) >> 20) as i32; /* extract exponent */
    if k == 0 {
        /* 0 or subnormal x */
        if lx | ((hx) & 0x7fffffff) == 0 {
            return x; /* +-0 */
        }
        x *= TWO54;
        //GET_HIGH_WORD(hx, x);
        hx = (x.repr() >> 32) as i32;
        k = (((hx & 0x7ff00000) >> 20) - 54) as i32;
        if n < -50000 {
            return TINY * x; /*underflow*/
        }
    }
    if k == 0x7ff {
        return x + x; /* NaN or Inf */
    }
    k = k + n as i32;
    if k > 0x7fe {
        return HUGE * copysign(HUGE, x); /* overflow  */
    }
    if k > 0 {
        /* normal result */
        let high_word:u64 = (((hx as u32)&0x800fffff)|((k<<20) as u32)) as u64;
        return f64::from_repr(high_word<<32 | (x.repr() & 0xffffffff));
    }
    if k <= -54 {
        if n > 50000 {
            /* in case integer overflow in n+k */
            return HUGE * copysign(HUGE, x); /*overflow*/
        } else {
            return TINY * copysign(TINY, x); /*underflow*/
        }
    }
    k += 54; /* subnormal result */

    let high_word:u64 = (((hx as u32)&0x800fffff)|((k<<20) as u32)) as u64;
    f64::from_repr(high_word<<32 | ((x.repr()) & 0xffffffff)) * TWOM54
}

pub extern "C" fn copysignf(x: f32, y: f32) -> f32 {
    let ix = x.repr();
    let iy = y.repr();
    f32::from_repr((ix & 0x7fffffff) | (iy & 0x80000000))
}

pub extern "C" fn copysign(x: f64, y: f64) -> f64 {
    f64::from_repr(
        (x.repr() & 0x7fffffffffffffff) | (y.repr() & 0x8000000000000000),
    )
}

#[cfg(test)]
mod more_tests {
    #[test]
    fn atanf() {
        use core::f32::consts::PI;
        assert_eq!(super::atanf(0f32), 0f32);
        assert_eq!(super::atanf(1f32), 0.7853982f32);
        assert_eq!(super::atanf(2f32), 1.1071488f32);
        assert_eq!(super::atanf(3f32), 1.2490457f32);
        assert_eq!(super::atanf(PI), 1.2626272f32);
    }

    #[test]
    fn cosf() {
        use core::f32::consts::PI;
        assert_eq!(super::cosf(0f32), 1f32);
        assert_eq!(super::cosf(0.1f32), 0.9950042f32);
        assert_eq!(super::cosf(0.2f32), 0.9800666f32);
        assert_eq!(super::cosf(0.3f32), 0.9553365f32);
        assert_eq!(super::cosf(0.4f32), 0.921061f32);
        assert_eq!(super::cosf(0.5f32), 0.87758255f32);
        assert_eq!(super::cosf(1f32), 0.5403023f32);
        assert_eq!(super::cosf(-1f32), 0.5403023f32);
        assert_eq!(super::cosf(2f32), -0.41614684f32);
        assert_eq!(super::cosf(3f32), -0.9899925f32);
        assert_eq!(super::cosf(4f32), -0.6536436f32);
        assert_eq!(super::cosf(PI), -1f32);
        assert_eq!(super::cosf(-PI), -1f32);
        assert_eq!(super::cosf(1f32), 0.540302305868f32);
    }

    #[test]
    fn scalbn() {
        assert_eq!(super::scalbn(0_f64, 0), 0_f64);
        assert_eq!(super::scalbn(-0_f64, 0), -0_f64);
        assert_eq!(super::scalbn(0.8_f64, 4), 12.8_f64);
        assert_eq!(super::scalbn(-0.854375_f64, 5), -27.34_f64);
        assert_eq!(super::scalbn(1_f64, 0), 1_f64);
        assert_eq!(super::scalbn(0.8_f64, 4), 12.8_f64);
        assert_eq!(super::scalbn(8718927381278827_f64, -18), 33260068440.547283_f64);
    }

    #[test]
    fn floorf() {
        assert_eq!(super::floorf(123.7f32), 123f32);
        assert_eq!(super::floorf(12345.6f32), 12345f32);
        assert_eq!(super::floorf(1.9f32), 1f32);
        assert_eq!(super::floorf(1.009f32), 1f32);
        assert_eq!(super::floorf(-1.009f32), -2f32);
        assert_eq!(super::floorf(-0.009f32), -1f32);
    }

    #[test]
    fn floor() {
        assert_eq!(super::floor(123.7f64), 123f64);
        assert_eq!(super::floor(12345.6f64), 12345f64);
        assert_eq!(super::floor(1.9f64), 1f64);
        assert_eq!(super::floor(1.009f64), 1f64);
        assert_eq!(super::floor(-1.009f64), -2f64);
        assert_eq!(super::floor(-0.009f64), -1f64);
        assert_eq!(super::floor(-14207526.52111395f64), -14207527f64);
    }

    #[test]
    fn sinf() {
        use core::f32::consts::PI;
        assert_eq!(super::sinf(0.00001f32), 0.00001f32);
        assert_eq!(super::sinf(0.001f32), 0.0009999999f32);
        assert_eq!(super::sinf(0.1f32), 0.09983342f32);
        assert_eq!(super::sinf(0.2f32), 0.19866933f32);
        assert_eq!(super::sinf(0.3f32), 0.29552022f32);
        assert_eq!(super::sinf(0.5f32), 0.47942555f32);
        assert_eq!(super::sinf(0.8f32), 0.7173561f32);
        assert_eq!(super::sinf(1f32), 0.84147096f32);
        assert_eq!(super::sinf(1.5f32), 0.997495f32);
        assert_eq!(super::sinf(2f32), 0.9092974f32);
        assert_eq!(super::sinf(0f32), 0f32);
        assert_eq!(super::sinf(-1f32), -0.84147096f32);
        assert_eq!(super::sinf(PI), -0.00000008742278_f32);
        assert_eq!(super::sinf(-PI), 0.00000008742278_f32);
        assert_eq!(super::sinf(2.5f32), 0.5984721_f32); 
        assert_eq!(super::sinf(2f32 * PI), 0.00000017484555f32); 
    }
}

#[cfg(test)]
check! {
// `atan` has not been implemented yet
// fn atan2(f: extern fn(f64, f64) -> f64, y: F64, x: F64) -> Option<F64> {
//     Some(F64(f(y.0, x.0)))
// }


fn atan2f(f: extern fn (f32, f32) -> f32, y: F32, x: F32) -> Option < F32 > {
Some(F32(f(y.0, x.0)))
}

fn atanf(f: extern fn (f32) -> f32, x: F32) -> Option <F32 > {
Some(F32(f(x.0)))
}

// unimplemented!
// fn atan(f: extern fn(f64) -> f64, x: F64) -> Option<F64> {
//     Some(F64(f(x.0)))
// }

fn fabs(f: extern fn (f64) -> f64, x: F64) -> Option <F64 > {
Some(F64(f(x.0)))
}

fn fabsf(f: extern fn (f32) -> f32, x: F32) -> Option <F32 > {
Some(F32(f(x.0)))
}

fn cosf(f: extern fn (f32) -> f32, x: F32limit) -> Option <F32limit > {
Some(F32limit(f(x.0)))
}

fn sinf(f: extern fn (f32) -> f32, x: F32limit) -> Option <F32limit > {
Some(F32limit(f(x.0)))
}

fn copysign(f: extern fn (f64, f64) -> f64, y: F64, x: F64) -> Option < F64 > {
Some(F64(f(y.0, x.0)))
}
fn copysignf(f: extern fn (f32, f32) -> f32, y: F32, x: F32) -> Option < F32 > {
Some(F32(f(y.0, x.0)))
}
fn scalbn(f: extern fn (f64, i32) -> f64, x: F64, n: i32) -> Option < F64 > {
Some(F64(f(x.0, n)))
}

fn floorf(f: extern fn (f32) -> f32, x: F32) -> Option <F32 > {
Some(F32(f(x.0)))
}
fn floor(f: extern fn (f64) -> f64, x: F64) -> Option <F64 > {
Some(F64(f(x.0)))
}

// unimplemented!
// fn sqrt(f: extern fn(f64) -> f64, x: F64) -> Option<F64> {
//     Some(F64(f(x.0)))
// }

fn sqrtf(f: extern fn (f32) -> f32, x: F32) -> Option <F32 > {
match () {
# [cfg(all(target_env = "gnu", target_os = "windows"))]
() => {
if x.0.repr() == 0x8000_0000 {
None
} else {
Some(F32(f(x.0)))
}
},
# [cfg(not(all(target_env = "gnu", target_os = "windows")))]
() => Some(F32(f(x.0))),
}

}
}
