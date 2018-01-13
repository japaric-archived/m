//! Bindings to libm (only for testing)

extern "C" {
    // pub fn atan(x: f64) -> f64;
    // pub fn atan2(y: f64, x: f64) -> f64;
    // pub fn sqrt(x: f64) -> f64;
    pub fn atan2f(y: f32, x: f32) -> f32;
    pub fn atanf(x: f32) -> f32;
    pub fn fabs(x: f64) -> f64;
    pub fn fabsf(x: f32) -> f32;
    pub fn sqrtf(x: f32) -> f32;
    pub fn cosf(x: f32) -> f32;
    pub fn sinf(x: f32) -> f32;
    pub fn copysign(x: f64, y: f64) -> f64;
    pub fn copysignf(x: f32, y: f32) -> f32;
    pub fn scalbn(x: f64, n: i32) -> f64;
    pub fn floorf(x: f32) -> f32;
    pub fn floor(x: f64) -> f64;
}
