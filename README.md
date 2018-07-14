# PSA: this crate has been deprecated in favor of [`libm`]

[`libm`]: https://github.com/japaric/libm

`libm` is a port of MUSL's libm. MUSL source code proved much easier to translate to Rust, and more
importantly it was easier to write a test generator that didn't require compiling C code and that
worked on a bunch of different architectures.

These two aspects make such a difference that we were able to [port 30 math functions to Rust in less
than 24 hours][libm-changelog].

[libm-changelog]: https://github.com/japaric/libm/blob/master/CHANGELOG.md#v011---2018-07-14

-@japaric, 2018-07-14

---

[![crates.io](https://img.shields.io/crates/d/m.svg)](https://crates.io/crates/m)
[![crates.io](https://img.shields.io/crates/v/m.svg)](https://crates.io/crates/m)

# `m`

> A C free / pure Rust mathematical library ("libm") for `no_std` code

This is a port of [OpenLibm].

[openlibm]: https://github.com/JuliaLang/openlibm

## [Documentation](https://docs.rs/m)

## [Change log](CHANGELOG.md)

## License

The m crate is a port of the OpenLibm library, which contains code that is
covered by various licenses:

The OpenLibm code derives from the FreeBSD msun and OpenBSD libm
implementations, which in turn derives from FDLIBM 5.3. As a result, it has a
number of fixes and updates that have accumulated over the years in msun, and
also optimized assembly versions of many functions. These improvements are
provided under the BSD and ISC licenses. The msun library also includes work
placed under the public domain, which is noted in the individual files. Further
work on making a standalone OpenLibm library from msun, as part of the Julia
project is covered under the MIT license.

TL;DR OpenLibm contains code that is licensed under the 2-clause BSD, the ISC
and the MIT licenses and code that is in the public domain. As a user of this
code you agree to use it under these licenses. As a contributor, you agree to
allow your code to be used under all these licenses as well.

Full text of the relevant licenses is in LICENSE.md.
