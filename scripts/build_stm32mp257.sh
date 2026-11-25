#!/usr/bin/env bash
# Cross-build CV-DAW C++20 DSP engine + lock-free IPC library for STM32MP257F-DK (Cortex-A35 AArch64).
#
# Run inside a Linux x86_64 host where the ST OpenSTLinux SDK is installed (e.g. /opt/st/...),
# OR natively on the STM32MP257F-DK board itself.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/realtime/build-stm32}"
VOICES="${CV_DAW_VOICES_PER_LIMB:-8}"
WITH_RUBBERBAND="${CV_DAW_WITH_RUBBERBAND:-OFF}"

CMAKE_ARGS=(
  -S "${ROOT_DIR}/realtime"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DCV_DAW_CORTEX_A35=ON
  -DCV_DAW_WITH_RUBBERBAND="${WITH_RUBBERBAND}"
  -DCV_DAW_VOICES_PER_LIMB="${VOICES}"
)

if [[ -n "${OECORE_NATIVE_SYSROOT:-}" ]]; then
  TOOLCHAIN="${OE_CMAKE_TOOLCHAIN_FILE:-${OECORE_NATIVE_SYSROOT}/usr/share/cmake/OEToolchainConfig.cmake}"
  if [[ ! -f "${TOOLCHAIN}" ]]; then
    echo "ERROR: OpenSTLinux toolchain file not found: ${TOOLCHAIN}" >&2
    exit 1
  fi
  echo "Cross-compiling with OpenSTLinux SDK toolchain: ${TOOLCHAIN}"
  CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}")
elif [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: Host is $(uname -m) and OpenSTLinux SDK environment is not sourced." >&2
  echo "Source your ST SDK first, e.g.:" >&2
  echo "  source /opt/st/stm32mp2/environment-setup-cortexa35-ostl-linux" >&2
  exit 1
else
  echo "Building natively on AArch64 host..."
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || echo 2)"

echo ""
echo "=== STM32MP257F-DK BUILD READY ==="
ls -lh "${BUILD_DIR}/cv-daw-engine" "${BUILD_DIR}/libcv_daw_ipc.so"
echo ""
echo "Deploy both files into <repo>/realtime/build/ on the STM32MP257F-DK board:"
echo "  scp ${BUILD_DIR}/cv-daw-engine ${BUILD_DIR}/libcv_daw_ipc.so root@<board-ip>:~/cv-daw/realtime/build/"
echo ""
echo "Verify DSP headroom on Cortex-A35 before live audio:"
echo "  ./cv-daw-engine --offline 2000 --demo --profile"
