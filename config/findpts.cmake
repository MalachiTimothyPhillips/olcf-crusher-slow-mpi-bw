set(FINDPTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/findpts)

set(FINDPTS_SOURCES
    ${FINDPTS_SOURCE_DIR}/ogsFindpts.cpp
    ${FINDPTS_SOURCE_DIR}/ogsFindptsLocal.cpp
    ${FINDPTS_SOURCE_DIR}/ogsFindpts.c
    ${FINDPTS_SOURCE_DIR}/ogsDevFindpts.c
    ${FINDPTS_SOURCE_DIR}/ogsHostFindpts.c
    ${FINDPTS_SOURCE_DIR}/ogsKernelsFindpts.cpp
)