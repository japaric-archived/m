set -ex

run() {
    local flags="--target $TARGET"

    cargo build $flags
    cargo build $flags --release

    case $TARGET in
        s390x-unknown-linux-gnu)
            cargo test $flags --no-run
            cargo test $flags --no-run --release
            ;;
        *)
            cargo test $flags
            cargo test $flags --release
            ;;
    esac
}

run
