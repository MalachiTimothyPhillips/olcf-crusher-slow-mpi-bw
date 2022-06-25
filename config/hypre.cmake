set(HYPRE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/hypre)
file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/include)
set(HYPRE_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/hypre)
set(HYPRE_INSTALL_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib/hypre")

ExternalProject_Add(
    HYPRE_BUILD
    SOURCE_DIR ${HYPRE_SOURCE_DIR}
    SOURCE_SUBDIR "src"
    BUILD_ALWAYS ON
    CMAKE_ARGS  -DHYPRE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                -DHYPRE_BUILD_TYPE=RelWithDebInfo
                -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
                -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
                -DHYPRE_ENABLE_SHARED=OFF
                -DHYPRE_ENABLE_MIXEDINT=ON
                -DHYPRE_ENABLE_SINGLE=OFF
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DCMAKE_C_VISIBILITY_PRESET=hidden
                -DCMAKE_CXX_VISIBILITY_PRESET=hidden
                -DCMAKE_INSTALL_LIBDIR=${HYPRE_INSTALL_LIB_DIR}
                -DCMAKE_INSTALL_INCLUDEDIR=${HYPRE_INSTALL_INCLUDE_DIR}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

add_library(nekrs-hypre-cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/elliptic/amgSolver/hypre/wrapper.c)
add_dependencies(nekrs-hypre-cpu HYPRE_BUILD)
target_include_directories(nekrs-hypre-cpu PRIVATE ${HYPRE_INSTALL_INCLUDE_DIR})
target_link_libraries(nekrs-hypre-cpu PRIVATE MPI::MPI_C ${HYPRE_INSTALL_LIB_DIR}/libHYPRE.a)
target_compile_definitions(nekrs-hypre-cpu PRIVATE HYPRE_API_PREFIX=NEKRS_)
set_target_properties(nekrs-hypre-cpu PROPERTIES LIBRARY_OUTPUT_NAME HYPRE)
install(TARGETS nekrs-hypre-cpu
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/hypre
  PRIVATE_HEADER DESTINATION ${HYPRE_INSTALL_INCLUDE_DIR} 
  PUBLIC_HEADER DESTINATION ${HYPRE_INSTALL_INCLUDE_DIR} 
)

if(ENABLE_CUDA)
  find_package(CUDAToolkit 10.0 REQUIRED)
  set(HYPRE_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/hypre/device)
  set(HYPRE_INSTALL_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/.hypre-device)

  ExternalProject_Add(
      HYPRE_BUILD_DEVICE
      SOURCE_DIR ${HYPRE_SOURCE_DIR}
      SOURCE_SUBDIR "src"
      BUILD_ALWAYS ON
      CMAKE_ARGS  -DHYPRE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                  -DHYPRE_BUILD_TYPE=RelWithDebInfo
                  -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
                  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
                  -DHYPRE_ENABLE_SHARED=OFF
                  -DHYPRE_ENABLE_MIXEDINT=ON
                  -DHYPRE_ENABLE_SINGLE=OFF
                  -DHYPRE_WITH_CUDA=${ENABLE_CUDA}
                  -DHYPRE_CUDA_SM=70
                  -DHYPRE_ENABLE_CUSPARSE=ON
                  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                  -DCMAKE_C_VISIBILITY_PRESET=hidden
                  -DCMAKE_CXX_VISIBILITY_PRESET=hidden
                  -DCMAKE_CUDA_VISIBILITY_PRESET=hidden
                  -DCMAKE_INSTALL_LIBDIR=${HYPRE_INSTALL_LIB_DIR}
                  -DCMAKE_INSTALL_INCLUDEDIR=${HYPRE_INSTALL_INCLUDE_DIR}
                  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  )

  add_library(nekrs-hypre-device SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/elliptic/amgSolver/hypre/wrapper.c)
  add_dependencies(HYPRE_BUILD_DEVICE HYPRE_BUILD)
  add_dependencies(nekrs-hypre-device HYPRE_BUILD_DEVICE)
  target_include_directories(nekrs-hypre-device PRIVATE ${HYPRE_INSTALL_INCLUDE_DIR})
  # TODO: get dependencies of external HYPRE project
  target_link_libraries(nekrs-hypre-device PRIVATE MPI::MPI_C CUDA::curand CUDA::cublas CUDA::cusparse CUDA::cusolver ${HYPRE_INSTALL_LIB_DIR}/libHYPRE.a)
  target_compile_definitions(nekrs-hypre-device PRIVATE HYPRE_API_PREFIX=NEKRS_DEVICE_)
  set_target_properties(nekrs-hypre-device PROPERTIES LIBRARY_OUTPUT_NAME HYPREDevice)
  set_target_properties(nekrs-hypre-device PROPERTIES LINK_OPTIONS ${BSYMBOLIC_FLAG})
  install(TARGETS nekrs-hypre-device
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/hypre
    PRIVATE_HEADER DESTINATION ${HYPRE_INSTALL_INCLUDE_DIR} 
    PUBLIC_HEADER DESTINATION ${HYPRE_INSTALL_INCLUDE_DIR} 
  )
endif()

set(HYPRE_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/hypre)
