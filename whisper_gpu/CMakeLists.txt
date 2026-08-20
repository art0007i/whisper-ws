cmake_minimum_required(VERSION 3.16)

project(whisper_wrapper LANGUAGES CXX)

include(FetchContent)

# ------------------------------------------------------------
# whisper.cpp configuration
# ------------------------------------------------------------

set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
set(GGML_NATIVE ON CACHE BOOL "Enable native CPU optimizations" FORCE)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# ------------------------------------------------------------
# whisper.cpp
# ------------------------------------------------------------

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG master
)

FetchContent_MakeAvailable(whisper)

# ------------------------------------------------------------
# Python wrapper
# ------------------------------------------------------------

add_library(whisper_wrapper SHARED
    whisper_wrapper.cpp
)

target_link_libraries(whisper_wrapper
    PRIVATE whisper
)

target_include_directories(whisper_wrapper
    PRIVATE ${whisper_SOURCE_DIR}/include
)

target_compile_features(whisper_wrapper PRIVATE cxx_std_17)

# ------------------------------------------------------------
# Put the wrapper next to CMakeLists.txt
# ------------------------------------------------------------

set_target_properties(whisper_wrapper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
