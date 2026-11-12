#!/bin/bash
set -euo pipefail
export LC_ALL=C LANG=C
run_dir="$HOME/gef-benchmark-runs/20261104-remaining"
extra_root=/dev/shm/testPF-gef-extra-20261104
deps_dir="$extra_root/deps"
export PATH="$HOME/.local/share/gef-toolchain/envs/build/bin:$deps_dir/bin:$PATH"
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
export PKG_CONFIG_PATH="$deps_dir/lib/pkgconfig"
export LD_LIBRARY_PATH="$deps_dir/lib:$HOME/.local/share/gef-toolchain/envs/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$deps_dir/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# Snappy's conda package omits its pkg-config description, as in the CI workflow.
cat > "$deps_dir/lib/pkgconfig/snappy.pc" <<PC
prefix=$deps_dir
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: snappy
Description: Snappy
Version: 1.2.2
Libs: -L\${libdir} -lsnappy
Cflags: -I\${includedir}
PC
plugin_flags=()
for plugin in BRIEFLZ BSC BZIP2 COPY CRUSH CSC DENSITY DOBOZ FARI FASTLZ GIPFELI LZF LZG LZHAM LZJB LZO MS-COMPRESS NCOMPRESS PITHY QUICKLZ WFLZ YALZ77 ZLIB ZLIB-NG ZLING ZPAQ; do
    plugin_flags+=("-DENABLE_${plugin}=no")
done
cmake -S "$extra_root/squash-0.7.0" -B "$extra_root/squash-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$extra_root/squash-install" \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_PREFIX_PATH="$deps_dir" -DCMAKE_C_FLAGS="-march=native -mtune=native -fcommon -I$deps_dir/include" \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native -I$deps_dir/include" \
    -DCMAKE_INSTALL_RPATH="$deps_dir/lib" "${plugin_flags[@]}" > "$run_dir/squash-configure.txt" 2>&1
cmake --build "$extra_root/squash-build" -j4 > "$run_dir/squash-build.txt" 2>&1
cmake --install "$extra_root/squash-build" > "$run_dir/squash-install.txt" 2>&1
cd "$HOME/gef-experiments"
# Apply only the existing CI compiler-compatibility fixes to a generated copy.
# LeCo and the main benchmark use the same GCC/libstdc++ ABI.
cp -a lib/Learn-to-Compress/headers "$extra_root/leco-headers"
sed -i '11,25s/^/\/\/ CI compatibility: /' "$extra_root/leco-headers/string/leco_uint256.h"
sed -i 's/register uint64_t/uint64_t/g' "$extra_root/leco-headers/popcount.h"
cat > "$run_dir/leco-cxx" <<WRAP
#!/bin/bash
exec "$HOME/.local/share/gef-toolchain/envs/build/bin/x86_64-conda-linux-gnu-g++" -I"$extra_root/leco-headers" "\$@"
WRAP
chmod +x "$run_dir/leco-cxx"
diff -u lib/Learn-to-Compress/headers/string/leco_uint256.h "$extra_root/leco-headers/string/leco_uint256.h" > "$run_dir/leco-traits-compat.patch" || test "$?" = 1
diff -u lib/Learn-to-Compress/headers/popcount.h "$extra_root/leco-headers/popcount.h" > "$run_dir/leco-register-compat.patch" || test "$?" = 1
squash_lib=$(find "$extra_root/squash-install" -name 'libsquash0.7.so' -print -quit)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEATS_WITH_SQUASH=ON -DNEATS_WITH_LECO=ON \
    -DSQUASH_INCLUDE_DIR="$extra_root/squash-install/include/squash-0.7" -DSQUASH_SHARED_LIB="$squash_lib" \
    -DCMAKE_PREFIX_PATH="$deps_dir" -DNEATS_ALP_HYBRID_CLANG=OFF \
    -DNEATS_ALP_CLANGXX="$run_dir/leco-cxx" \
    -DNEATS_LECO_EXTRA_INCLUDES="$deps_dir/include;$deps_dir/include/eigen3" \
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE -DBUILD_TESTING=ON > "$run_dir/configure.txt" 2>&1
cmake --build build --target LosslessBenchmarkFull -j4 > "$run_dir/build.txt" 2>&1
ctest --test-dir build --output-on-failure > "$run_dir/tests.txt" 2>&1
./build/LosslessBenchmarkFull -h > "$run_dir/available.txt" 2>&1
cp build/CMakeCache.txt "$run_dir/CMakeCache.txt"
git diff --binary > "$run_dir/tracked-source.patch"
git submodule status > "$run_dir/submodules.txt"
sha256sum "$extra_root/squash-0.7.0.tar.bz2" > "$run_dir/squash-source.sha256"
cp -a "$deps_dir/conda-meta" "$run_dir/dependency-manifest"
echo 0 > "$run_dir/build.exit"
