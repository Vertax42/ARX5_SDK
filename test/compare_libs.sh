#!/usr/bin/env bash
# compare_libs.sh — Build test binaries against original prebuilt .so files and
# newly rebuilt source-based .so files, run both, and diff the numeric output.
#
# Usage:
#   cd /home/vertax/ARX5_SDK
#   bash test/compare_libs.sh [--build-new]
#
# Pass --build-new to (re)build the source-based libraries before comparing.
# Without it the script assumes build_lerobot/ already exists.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_NEW=0
for arg in "$@"; do [[ "$arg" == "--build-new" ]] && BUILD_NEW=1; done

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ORIG_LIB="$REPO/lib/x86_64"
NEW_BUILD="$REPO/build_lerobot"
NEW_LIB="$NEW_BUILD"
TMP="$REPO/test/tmp_compare"
mkdir -p "$TMP"

# ---------------------------------------------------------------------------
# Conda / compiler setup
# ---------------------------------------------------------------------------
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: activate the lerobot-xense conda environment first."
    exit 1
fi
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2"
INCLUDES="-I$REPO/include -I$CONDA_PREFIX/include -I$CONDA_PREFIX/include/urdfdom_headers -I$CONDA_PREFIX/include/kdl_parser"
EIGEN_INC="$(pkg-config --cflags eigen3 2>/dev/null || echo "-I$CONDA_PREFIX/include/eigen3")"

# ---------------------------------------------------------------------------
# Optionally rebuild from source
# ---------------------------------------------------------------------------
if [[ $BUILD_NEW -eq 1 ]]; then
    echo "=== Building source-based libraries ==="
    mkdir -p "$NEW_BUILD"
    cmake -S "$REPO" -B "$NEW_BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
        -DCMAKE_CXX_STANDARD=17
    cmake --build "$NEW_BUILD" --target hardware solver -j"$(nproc)"
fi

# ---------------------------------------------------------------------------
# Verify the necessary .so files exist
# ---------------------------------------------------------------------------
for f in "$ORIG_LIB/libhardware.so" "$ORIG_LIB/libsolver.so"; do
    [[ -f "$f" ]] || { echo "ERROR: original library not found: $f"; exit 1; }
done
for f in "$NEW_LIB/libhardware.so" "$NEW_LIB/libsolver.so"; do
    [[ -f "$f" ]] || { echo "ERROR: new library not found: $f (run with --build-new)"; exit 1; }
done

# ---------------------------------------------------------------------------
# Helper: compile a test against a specific lib dir
# ---------------------------------------------------------------------------
compile_test() {
    local src="$1"      # e.g. test/test_math_ops.cpp
    local libdir="$2"   # e.g. lib/x86_64 or build_lerobot
    local outbin="$3"   # output binary path

    $CXX $CXXFLAGS $INCLUDES $EIGEN_INC \
        "$src" \
        -L"$libdir" -L"$CONDA_PREFIX/lib" \
        -lhardware -lsolver -lkdl_parser -lorocos-kdl \
        -Wl,-rpath,"$libdir" \
        -Wl,-rpath,"$CONDA_PREFIX/lib" \
        -o "$outbin"
}

# ---------------------------------------------------------------------------
# Run one comparison for a given test source
# ---------------------------------------------------------------------------
run_comparison() {
    local name="$1"
    local src="$REPO/test/${name}.cpp"
    local bin_orig="$TMP/${name}_orig"
    local bin_new="$TMP/${name}_new"
    local out_orig="$TMP/${name}_orig.txt"
    local out_new="$TMP/${name}_new.txt"
    local diff_file="$TMP/${name}.diff"

    echo ""
    echo "==================================================================="
    echo "  Test: $name"
    echo "==================================================================="

    echo "  [1/4] Compiling against original lib/x86_64/ ..."
    if ! compile_test "$src" "$ORIG_LIB" "$bin_orig" 2>"$TMP/${name}_orig_build.log"; then
        echo "  COMPILE ERROR (original). See: $TMP/${name}_orig_build.log"
        cat "$TMP/${name}_orig_build.log"
        return 1
    fi

    echo "  [2/4] Compiling against new $NEW_LIB/ ..."
    if ! compile_test "$src" "$NEW_LIB" "$bin_new" 2>"$TMP/${name}_new_build.log"; then
        echo "  COMPILE ERROR (new). See: $TMP/${name}_new_build.log"
        cat "$TMP/${name}_new_build.log"
        return 1
    fi

    echo "  [3/4] Running original binary ..."
    set +e
    "$bin_orig" >"$out_orig" 2>&1
    orig_exit=$?
    set -e
    if [[ $orig_exit -ne 0 ]]; then
        echo "  NOTE: original binary exited with code $orig_exit (internal test failures counted below)"
    fi

    echo "  [4/4] Running new binary ..."
    set +e
    "$bin_new" >"$out_new" 2>&1
    new_exit=$?
    set -e
    if [[ $new_exit -ne 0 ]]; then
        echo "  NOTE: new binary exited with code $new_exit (internal test failures counted below)"
    fi

    # Strip lines with timestamps (e.g. kdl_parser [WARN] lines vary between runs)
    # Also strip lines with non-deterministic values (multi_trial_ik q_out varies by RNG seed)
    grep -v '^\[' "$out_orig" \
        | grep -v '^ *q_out' \
        | grep -v 'multi_trial_ik pos_err' \
        >"${out_orig}.filtered" || true
    grep -v '^\[' "$out_new" \
        | grep -v '^ *q_out' \
        | grep -v 'multi_trial_ik pos_err' \
        >"${out_new}.filtered"  || true

    if diff -u "${out_orig}.filtered" "${out_new}.filtered" >"$diff_file" 2>&1; then
        echo "  RESULT: IDENTICAL — outputs match exactly."
        return 0
    else
        local nlines
        nlines=$(wc -l <"$diff_file")
        echo "  RESULT: DIFF FOUND ($nlines diff lines). See: $diff_file"
        echo "  --- first 40 diff lines ---"
        head -40 "$diff_file"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "==================================================================="
echo "  ARX5 Library Equivalence Test"
echo "  Original: $ORIG_LIB"
echo "  New:      $NEW_LIB"
echo "==================================================================="

PASS=0; FAIL=0

run_comparison "test_math_ops" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_comparison "test_solver"   && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo "==================================================================="
printf "  SUMMARY: %d passed, %d failed\n" "$PASS" "$FAIL"
echo "==================================================================="

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
