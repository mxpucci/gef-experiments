#!/usr/bin/env bash
# Run on the x86-64 destination. All tools stay within the user's account.
set -euo pipefail
export LC_ALL=C LANG=C
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolchain_root="${GEF_TOOLCHAIN_ROOT:-$HOME/.local/share/gef-toolchain}"
mamba_bin="${GEF_MICROMAMBA:-$HOME/.local/bin/micromamba}"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "This helper provisions a Linux x86-64 toolchain; use normal CMake on other hosts." >&2
    exit 1
fi
if [[ ! -x "$mamba_bin" ]]; then
    mkdir -p "$(dirname "$mamba_bin")"
    download_dir="$(mktemp -d)"
    trap 'rm -rf "$download_dir"' EXIT
    curl -fLsS https://micro.mamba.pm/api/micromamba/linux-64/2.10.0 -o "$download_dir/micromamba.tar.bz2"
    tar -xjf "$download_dir/micromamba.tar.bz2" -C "$download_dir" bin/micromamba
    cp "$download_dir/bin/micromamba" "$mamba_bin"
fi
if [[ ! -x "$toolchain_root/envs/build/bin/x86_64-conda-linux-gnu-g++" ||
      ! -x "$toolchain_root/envs/build/bin/rustc" ]]; then
    MAMBA_ROOT_PREFIX="$toolchain_root" "$mamba_bin" create -y -n build -c conda-forge \
        --strict-channel-priority gcc_linux-64=13.4.0 gxx_linux-64=13.4.0 rust=1.87
fi
export PATH="$toolchain_root/envs/build/bin:$PATH"
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
# CentOS 7 ships Git 1.8.3, before git -C was introduced.
(cd "$repo_dir" && git submodule update --init --recursive)
cmake -S "$repo_dir" -B "$repo_dir/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DNEATS_WITH_SQUASH=OFF -DNEATS_WITH_LECO=OFF -DNEATS_WITH_BUFF=ON \
    -DNEATS_ALP_HYBRID_CLANG=OFF -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE \
    -DGEF_WITH_PCODEC=ON -DGEF_WITH_FAUNA=ON -DGEF_WITH_FLEA=ON -DBUILD_TESTING=ON
cmake --build "$repo_dir/build" -j "${GEF_BUILD_JOBS:-4}"
ctest --test-dir "$repo_dir/build" --output-on-failure
