use std::{f32, f64, fmt};

use quickcheck::{Arbitrary, Gen};

use {FloatExt, Sign};

macro_rules! check {
    ($(
        fn $name:ident($f:ident: extern fn($($farg:ty),+) -> $fret:ty,
                       $($arg:ident: $t:ty),+) -> Option<$ret:ty> {
            $($code:tt)*
        }
    )+) => {
        mod tests {
            use std::mem;

            use quickcheck::TestResult;

            use FloatExt;
            use qc::*;

            $(
                #[test]
                fn $name() {
                    fn check($($arg: $t),+) -> TestResult {
                        fn $name($f: extern fn($($farg),+) -> $fret,
                                 $($arg:$t),+)
                                 -> Option<$ret> {
                            $($code)*
                        }

                        let our_answer = $name(super::$name, $($arg),+);
                        let libm_f: unsafe extern "C" fn($($farg),+) -> $fret =
                            ::m::$name;
                        let libm_answer =
                            $name(unsafe { mem::transmute(libm_f) }, $($arg),+);

                        if our_answer.is_none() {
                            return TestResult::discard();
                        }

                        let our_answer = our_answer.unwrap();
                        let libm_answer = libm_answer.unwrap();

                        let print_values = || {
                            print!("\r{} - Args: ", stringify!($name));
                            $(print!("{} = {:?} ", stringify!($arg), $arg);)+
                                print!("\n");
                            println!("  us:   {:?}", our_answer);
                            println!("  libm: {:?}", libm_answer);
                        };

                        if !our_answer.0.eq_repr(libm_answer.0) {
                            print_values();
                            TestResult::from_bool(false)
                        } else {
                            TestResult::from_bool(true)
                        }
                    }

                    ::quickcheck::quickcheck(check as fn($($t),*) -> TestResult)
                }
            )+
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct F32(pub f32);

impl fmt::Debug for F32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (0x{:08x})", self.0, self.0.repr())
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct F64(pub f64);

impl fmt::Debug for F64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (0x{:016x})", self.0, self.0.repr())
    }
}

macro_rules! arbitrary_float {
    ($ty:ident, $fty:ident) => {
        impl Arbitrary for $ty {
            fn arbitrary<G>(g: &mut G) -> $ty
                where G: Gen
            {
                let special =
                    [-0.0, 0.0, $fty::NAN, $fty::INFINITY, $fty::NEG_INFINITY];

                let (sign, mut exponent, mut significand) = g.gen();
                if g.gen_weighted_bool(10) {
                    return $ty(*g.choose(&special).unwrap());
                } else if g.gen_weighted_bool(10) {
                    // NaN variants
                    significand = 0;
                } else if g.gen() {
                    // denormalize
                    exponent = 0;
                }

                $ty($fty::from_parts(Sign::from_bool(sign),
                                     exponent,
                                     significand))
            }
        }
    }
}

arbitrary_float!(F32, f32);
arbitrary_float!(F64, f64);
