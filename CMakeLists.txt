cmake_minimum_required(VERSION 3.16)
project(idcol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)

# ---------- Core sources ----------
set(IDCOL_CORE
    core/idcol_solve.cpp
    core/idcol_newton.cpp
    core/idcol_kkt.cpp
    core/shape_core.cpp
    core/radial_bounds.cpp
)

# ---------- Helper to register an example executable ----------
function(add_idcol_example tgt src)
    add_executable(${tgt} ${src} ${IDCOL_CORE})

    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${tgt} PRIVATE Eigen3::Eigen)

    if (MSVC)
        target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    else()
        target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)
    endif()

    target_compile_definitions(${tgt} PRIVATE
        $<$<CONFIG:Release>:EIGEN_NO_DEBUG EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE>
    )
endfunction()

# ---------- Executables ----------
add_idcol_example(idcol_example  examples/main.cpp)
add_idcol_example(idcol_ergodic  examples/ergodic.cpp)
add_idcol_example(idcol_scaling  examples/scaling_polytope.cpp)