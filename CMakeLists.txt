cmake_minimum_required(VERSION 3.16)
project(idcol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Force Release build by default
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release by default." FORCE)
    message(STATUS "Defaulting to Release build for maximum performance.")
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

# ---------- Core sources ----------
set(IDCOL_CORE
    core/idcol_solve.cpp
    core/idcol_newton.cpp
    core/idcol_kkt.cpp
    core/shape_core.cpp
    core/radial_bounds.cpp
)

# 2. Build core as a STATIC library
# This drastically reduces compile time (core is compiled once, not 3 times)
# and STATIC allows the compiler to optimize the executables more aggressively than SHARED.
add_library(idcol_core STATIC ${IDCOL_CORE})

target_include_directories(idcol_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(idcol_core PUBLIC Eigen3::Eigen)

# 3. Apply aggressive optimizations directly to the target
if (MSVC)
    target_compile_options(idcol_core PUBLIC $<$<CONFIG:Release>:/O2 /DNDEBUG /fp:fast>)
else()
    # Added -ffast-math for floating-point heavy computations
    target_compile_options(idcol_core PUBLIC $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math>)
endif()

target_compile_definitions(idcol_core PUBLIC
    $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE>
)

# 4. Enable Interprocedural Optimization (LTO)
# Allows the compiler to inline Eigen operations 
# and custom math across different .cpp files.
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    message(STATUS "IPO / LTO enabled: compiler will optimize across translation units.")
    set_target_properties(idcol_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ---------- Helper to register an example executable ----------
function(add_idcol_example tgt src)
    add_executable(${tgt} ${src})
    
    # Link the optimized static core
    target_link_libraries(${tgt} PRIVATE idcol_core)

    # Enable LTO for the executable as well
    if(ipo_supported)
        set_target_properties(${tgt} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

# ---------- Executables ----------
add_idcol_example(idcol_example  examples/main.cpp)
add_idcol_example(idcol_ergodic  examples/ergodic.cpp)
add_idcol_example(idcol_scaling  examples/scaling_polytope.cpp)