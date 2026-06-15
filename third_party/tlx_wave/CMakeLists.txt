set(TLX_WAVE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../wave")
set(TLX_WAVE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/wave")
set(TLX_WAVE_LLVM_INSTALL_DIR "${TLX_WAVE_SOURCE_DIR}/build/llvm-install")

if(NOT EXISTS "${TLX_WAVE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "tlx_wave requires the third_party/wave submodule. "
    "Run `git submodule update --init third_party/wave`.")
endif()

function(tlx_wave_mlir_has_python_bindings LLVM_INSTALL_DIR OUT_VAR)
  set(_mlir_config "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
  if(EXISTS "${_mlir_config}")
    file(STRINGS "${_mlir_config}" _mlir_python_bindings
      REGEX "set\\(MLIR_ENABLE_BINDINGS_PYTHON[ \t]+\"?ON\"?\\)")
    if(_mlir_python_bindings)
      set(${OUT_VAR} TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
  set(${OUT_VAR} FALSE PARENT_SCOPE)
endfunction()

macro(tlx_wave_select_llvm_install LLVM_INSTALL_PATH)
  set(LLVM_INSTALL_DIR "${LLVM_INSTALL_PATH}" CACHE PATH
    "LLVM/MLIR install prefix used by the Wave submodule" FORCE)
  set(LLVM_DIR "${LLVM_INSTALL_DIR}/lib/cmake/llvm" CACHE PATH
    "LLVM package directory used by the Wave submodule" FORCE)
  set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH
    "MLIR package directory used by the Wave submodule" FORCE)
  set(Clang_DIR "${LLVM_INSTALL_DIR}/lib/cmake/clang" CACHE PATH
    "Clang package directory used by the Wave submodule" FORCE)
  set(LLD_DIR "${LLVM_INSTALL_DIR}/lib/cmake/lld" CACHE PATH
    "LLD package directory used by the Wave submodule" FORCE)
endmacro()

set(TLX_WAVE_AUTO_BUILD_LLVM ON CACHE BOOL
  "Run the Wave LLVM builder when tlx_wave cannot find MLIR Python bindings.")

if(NOT DEFINED LLVM_INSTALL_DIR OR "${LLVM_INSTALL_DIR}" STREQUAL "")
  if(EXISTS "${TLX_WAVE_LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
    tlx_wave_select_llvm_install("${TLX_WAVE_LLVM_INSTALL_DIR}")
  elseif(NOT "${LLVM_SYSPATH}" STREQUAL "")
    tlx_wave_select_llvm_install("${LLVM_SYSPATH}")
  elseif(DEFINED LLVM_LIBRARY_DIR AND NOT "${LLVM_LIBRARY_DIR}" STREQUAL "")
    cmake_path(GET LLVM_LIBRARY_DIR PARENT_PATH _tlx_wave_llvm_install_dir)
    tlx_wave_select_llvm_install("${_tlx_wave_llvm_install_dir}")
  endif()
endif()

tlx_wave_mlir_has_python_bindings("${LLVM_INSTALL_DIR}" _tlx_wave_has_mlir_python)
if(NOT _tlx_wave_has_mlir_python)
  tlx_wave_mlir_has_python_bindings("${LLVM_SYSPATH}" _tlx_wave_syspath_has_mlir_python)
  tlx_wave_mlir_has_python_bindings("${TLX_WAVE_LLVM_INSTALL_DIR}" _tlx_wave_local_has_mlir_python)
  if(_tlx_wave_syspath_has_mlir_python)
    tlx_wave_select_llvm_install("${LLVM_SYSPATH}")
  elseif(_tlx_wave_local_has_mlir_python)
    tlx_wave_select_llvm_install("${TLX_WAVE_LLVM_INSTALL_DIR}")
  elseif(TLX_WAVE_AUTO_BUILD_LLVM)
    if(TRITON_OFFLINE_BUILD)
      message(FATAL_ERROR
        "tlx_wave requires an LLVM/MLIR install with MLIR Python bindings. "
        "Run `python third_party/wave/build_tools/build_llvm.py --python-bindings` "
        "or set LLVM_SYSPATH to a compatible install before configuring offline.")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(_tlx_wave_build_llvm_args
      "${Python3_EXECUTABLE}"
      "${TLX_WAVE_SOURCE_DIR}/build_tools/build_llvm.py"
      "--python-bindings")
    if(DEFINED ENV{MAX_JOBS} AND NOT "$ENV{MAX_JOBS}" STREQUAL "")
      list(APPEND _tlx_wave_build_llvm_args "--jobs" "$ENV{MAX_JOBS}")
    endif()
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}" -E env "LLVM_INSTALL_DIR="
        ${_tlx_wave_build_llvm_args}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
      COMMAND_ERROR_IS_FATAL ANY)
    tlx_wave_select_llvm_install("${TLX_WAVE_LLVM_INSTALL_DIR}")
  else()
    message(FATAL_ERROR
      "tlx_wave requires MLIR Python bindings, but LLVM_INSTALL_DIR='${LLVM_INSTALL_DIR}' "
      "does not have MLIR_ENABLE_BINDINGS_PYTHON=ON.")
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

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTLXWave ${CMAKE_CURRENT_SOURCE_DIR}/triton_tlx_wave.cc)
  target_link_libraries(TritonTLXWave PRIVATE Python3::Module pybind11::headers)
endif()

add_subdirectory("${TLX_WAVE_SOURCE_DIR}" "${TLX_WAVE_BINARY_DIR}")

set(TLX_WAVE_RUNTIME_TARGETS
  WavePythonModules
  wave-calibrate-report
  wave-opt
  wave-sim-report
  wave-symbols-test
  wave-translate
  wavec)

add_custom_target(TLXWaveSubmoduleRuntime ALL
  DEPENDS ${TLX_WAVE_RUNTIME_TARGETS})
