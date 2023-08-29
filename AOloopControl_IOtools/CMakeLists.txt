set(LIBNAME "cacaoAOloopControlIOtools")
set(SRCNAME "AOloopControl_IOtools")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")


set(SOURCEFILES
	${SRCNAME}.c
    ${SRCNAME}.h
    acquireWFSim.c
    findspots.c
    WFScamsim.c
    WFSmap.c
    shs_on_gpu/shs_gpu_ref.c
    shs_on_gpu/ref_recorder/SGR_Recorder.cu
    shs_on_gpu/ref_recorder/SGR_Recorder_interface.cpp
    shs_on_gpu/ref_recorder/SGR_ImageHandlerBase.cpp
    shs_on_gpu/ref_recorder/SGR_ImageHandler.cpp
    shs_on_gpu/util/SpotFitter.cpp
    shs_on_gpu/util/OneDGaussianFit.cpp
    shs_on_gpu/shs_gpu.c
    shs_on_gpu/evaluator/SGE_Evaluator.cu
    shs_on_gpu/evaluator/SGE_Evaluator_interface.cpp
    shs_on_gpu/evaluator/SGE_GridLayout.cu
    shs_on_gpu/evaluator/SGE_ReferenceManager.cpp
)

set(INCLUDEFILES
	${SRCNAME}.h
    acquireWFSim.h
    findspots.h
    AOloopControl_IOtools.h
    WFScamsim.h
    WFSmap.h
    shs_on_gpu/shs_gpu_ref.h
    shs_on_gpu/ref_recorder/SGR_Recorder.hpp
    shs_on_gpu/ref_recorder/SGR_Recorder_interface.h
    shs_on_gpu/ref_recorder/SGR_ImageHandlerBase.hpp
    shs_on_gpu/ref_recorder/SGR_ImageHandler.hpp
    shs_on_gpu/util/atypeUtil.hpp
    shs_on_gpu/util/Point.hpp
    shs_on_gpu/util/Rectangle.hpp
    shs_on_gpu/util/SpotFitter.hpp
    shs_on_gpu/util/OneDGaussianFit.hpp
    shs_on_gpu/util/CudaSGEutil.hpp
    shs_on_gpu/shs_gpu.h
    shs_on_gpu/evaluator/SGE_Evaluator.hpp
    shs_on_gpu/evaluator/SGE_Evaluator_interface.h
    shs_on_gpu/evaluator/SGE_GridLayout.hpp
    shs_on_gpu/evaluator/SGE_Kernel.hpp
    shs_on_gpu/evaluator/SGE_ReferenceManager.hpp
)


set(SCRIPTS
    scripts/cacao-avg-stream
    scripts/cacao-shs-gpu-ref
    scripts/cacao-shs-on-gpu
)

set(LINKLIBS
	CLIcore
    milkinfo
    cacaoAOloopControl
)



# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")

# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/..
	${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

install(PROGRAMS ${SCRIPTS} DESTINATION bin)









#project(lib_${LIBNAME}_project)

#add_library(${LIBNAME} SHARED ${SRCNAME}.c
#${SRCNAME}.h
#AOloopControl_IOtools_camerainput.c
#AOloopControl_IOtools_datastream_processing.c
#AOloopControl_IOtools_load_image_sharedmem.c
#AOloopControl_IOtools_RTLOGsave.c
#)

#target_link_libraries(${LIBNAME} PUBLIC CLIcore milkinfo cacaoAOloopControl)

#install(TARGETS ${LIBNAME} DESTINATION lib)
#install(FILES ${SRCNAME}.h DESTINATION include/${SRCNAME})
