set -ex

main() {
    cross generate-lockfile

    cross build --target $TARGET
    cross build --target $TARGET --release

    if [ $TARGET != s390x-unknown-linux-gnu ]; then
        cross test --target $TARGET
        cross test --target $TARGET --release
    fi
}

main
