cmake_minimum_required( VERSION 3.16.3 )

project( arx5-sdk )

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # global PIC

# Prefer conda-managed dependencies over system packages.
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

# Use spdlog's bundled fmt. MUST be set before find_package(spdlog).
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "Use spdlog bundled fmt, not system fmt" FORCE)

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(spdlog REQUIRED)

# ---------------------------------------------------------------------------
# RPATH: embed conda lib dir so the built .so files find spdlog at runtime
# without needing LD_LIBRARY_PATH.
# ---------------------------------------------------------------------------
set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_BUILD_RPATH   "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_compile_definitions(SDK_ROOT="${CMAKE_BINARY_DIR}/..")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/aarch64)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(
    $ENV{CONDA_PREFIX}/include/
    $ENV{CONDA_PREFIX}/include/urdfdom_headers
)

# Add conda lib directory to link directories
link_directories($ENV{CONDA_PREFIX}/lib)

# ---------------------------------------------------------------------------
# libhardware: CAN hardware abstraction (SocketCAN + EtherCAT)
# ---------------------------------------------------------------------------
add_library(hardware SHARED
    src/hardware/math_ops.cpp
    src/libcan/SocketCAN.cpp
    src/hardware/arx_can.cpp
)
target_include_directories(hardware PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    $ENV{CONDA_PREFIX}/include/soem
)
target_link_libraries(hardware
    Threads::Threads
    soem
)

# ---------------------------------------------------------------------------
# libsolver: KDL-based kinematics / dynamics solver
# ---------------------------------------------------------------------------
add_library(solver SHARED
    src/app/solver.cpp
)
target_include_directories(solver PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    $ENV{CONDA_PREFIX}/include/kdl_parser
    $ENV{CONDA_PREFIX}/include/urdfdom_headers
)
target_link_libraries(solver
    Eigen3::Eigen
    kdl_parser::kdl_parser
    orocos-kdl
)

add_library(ArxJointController SHARED
    src/app/joint_controller.cpp
    src/app/controller_base.cpp
    src/utils.cpp
)
target_link_libraries(ArxJointController
    hardware
    solver
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    kdl_parser::kdl_parser
    orocos-kdl
    soem
)
add_library(ArxCartesianController SHARED
    src/app/cartesian_controller.cpp
    src/app/controller_base.cpp
    src/utils.cpp
)
target_link_libraries(ArxCartesianController
    hardware
    solver
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    kdl_parser::kdl_parser
    orocos-kdl
    soem
)

# Hack for py310 (conda environment is slightly different from other python versions)
target_include_directories(ArxJointController PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser  $ENV{CONDA_PREFIX}/include/urdfdom_headers)
target_include_directories(ArxCartesianController PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser  $ENV{CONDA_PREFIX}/include/urdfdom_headers)


add_executable(test_joint_controller examples/test_joint_controller.cpp)
target_link_libraries(test_joint_controller
    hardware
    solver
    ArxJointController
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    kdl_parser::kdl_parser
    orocos-kdl
    soem
)

add_executable(test_cartesian_controller examples/test_cartesian_controller.cpp)
target_link_libraries(test_cartesian_controller
    hardware
    solver
    ArxCartesianController
    spdlog::spdlog
    Eigen3::Eigen
    Threads::Threads
    kdl_parser::kdl_parser
    orocos-kdl
    soem
)

# ---------------------------------------------------------------------------
# Unit tests (math_ops and solver)
# ---------------------------------------------------------------------------
add_executable(test_math_ops test/test_math_ops.cpp)
target_link_libraries(test_math_ops
    hardware
)

add_executable(test_solver test/test_solver.cpp)
target_link_libraries(test_solver
    solver
    Eigen3::Eigen
)

add_executable(test_can_encoding test/test_can_encoding.cpp)
target_link_libraries(test_can_encoding
    hardware
)

add_executable(test_can_vcan test/test_can_vcan.cpp)
target_link_libraries(test_can_vcan
    hardware
    Threads::Threads
)

enable_testing()
add_test(NAME math_ops      COMMAND test_math_ops)
add_test(NAME solver        COMMAND test_solver
    ${CMAKE_SOURCE_DIR}/models/X5.urdf)
add_test(NAME can_encoding  COMMAND test_can_encoding)

add_subdirectory(python)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)

install(FILES ${LIB_DIR}/libhardware.so ${LIB_DIR}/libsolver.so
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)

install(TARGETS ArxJointController ArxCartesianController
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)
