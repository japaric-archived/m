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
}
