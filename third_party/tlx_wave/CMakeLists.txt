set(TLX_WAVE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../wave")
set(TLX_WAVE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/wave")
set(TLX_WAVE_LLVM_INSTALL_DIR "${TLX_WAVE_SOURCE_DIR}/build/llvm-install")

if(NOT EXISTS "${TLX_WAVE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "tlx_wave requires the third_party/wave submodule. "
    "Run `git submodule update --init third_party/wave`.")
endif()

if(NOT DEFINED LLVM_INSTALL_DIR OR "${LLVM_INSTALL_DIR}" STREQUAL "")
  if(EXISTS "${TLX_WAVE_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
    set(LLVM_INSTALL_DIR "${TLX_WAVE_LLVM_INSTALL_DIR}" CACHE PATH
      "LLVM/MLIR install prefix used by the Wave submodule" FORCE)
  elseif(NOT "${LLVM_SYSPATH}" STREQUAL "")
    set(LLVM_INSTALL_DIR "${LLVM_SYSPATH}" CACHE PATH
      "LLVM/MLIR install prefix used by the Wave submodule" FORCE)
  elseif(DEFINED LLVM_LIBRARY_DIR AND NOT "${LLVM_LIBRARY_DIR}" STREQUAL "")
    cmake_path(GET LLVM_LIBRARY_DIR PARENT_PATH _tlx_wave_llvm_install_dir)
    set(LLVM_INSTALL_DIR "${_tlx_wave_llvm_install_dir}" CACHE PATH
      "LLVM/MLIR install prefix used by the Wave submodule" FORCE)
  endif()
endif()

if(NOT DEFINED WAVE_LLVM_PROJECT_SRC_DIR OR "${WAVE_LLVM_PROJECT_SRC_DIR}" STREQUAL "")
  set(WAVE_LLVM_PROJECT_SRC_DIR "${TLX_WAVE_SOURCE_DIR}/build/_deps/llvm-project" CACHE PATH
    "llvm-project source directory used by the Wave submodule" FORCE)
endif()

if(NOT DEFINED WAVE_LLVM_PROJECT_BUILD_DIR OR "${WAVE_LLVM_PROJECT_BUILD_DIR}" STREQUAL "")
  set(WAVE_LLVM_PROJECT_BUILD_DIR "${TLX_WAVE_SOURCE_DIR}/build/llvm-build" CACHE PATH
    "llvm-project build directory used by the Wave submodule" FORCE)
endif()

set(WAVE_ENABLE_PYTHON_BINDINGS ON CACHE BOOL
  "Build the Wave/WaveAMD MLIR Python bindings for tlx_wave." FORCE)

add_subdirectory("${TLX_WAVE_SOURCE_DIR}" "${TLX_WAVE_BINARY_DIR}")

add_custom_target(TLXWaveSubmoduleRuntime ALL
  DEPENDS
    wave-opt
    WavePythonModules)
