set(LIBNAME "cacaoAOloopControlIOtools")
set(SRCNAME "AOloopControl_IOtools")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")


set(SOURCEFILES
	${SRCNAME}.c
    ${SRCNAME}.h
    acquireWFSim.c
    acquireWFSspec.c
    ao188_preprocessor.c
    findspots.c
    shs_on_gpu/evaluator/SGE_Evaluator.cu
    shs_on_gpu/evaluator/SGE_Evaluator_interface.cpp
    shs_on_gpu/evaluator/SGE_GridLayout.cu
    shs_on_gpu/evaluator/SGE_ReferenceManager.cu
    shs_on_gpu/pupil_reshape.c
    shs_on_gpu/pupil_reshaper/SGE_Reshaper.cpp
    shs_on_gpu/pupil_reshaper/SGE_Reshaper_interface.cpp
    shs_on_gpu/pupil_zern_generator/PupilZernikeGenerator_interface.cpp
    shs_on_gpu/pupil_zern_generator/PupilZernikeGenerator.cpp
    shs_on_gpu/pupil_zernike_gen.c
    shs_on_gpu/ref_recorder/SGR_Recorder.cu
    shs_on_gpu/ref_recorder/SGR_Recorder_interface.cpp
    shs_on_gpu/shs_gpu.c
    shs_on_gpu/shs_gpu_ref.c
    shs_on_gpu/util/Circle.cpp
    shs_on_gpu/util/GaussianKernel.cu
    shs_on_gpu/util/ImageHandler2DBase.cu
    shs_on_gpu/util/OneDGaussianFit.cpp
    shs_on_gpu/util/Pupil.cpp
    shs_on_gpu/util/SpotFitter.cpp
    shs_on_gpu/wf_reconstructor/modalwfreconstructor.cpp
    shs_on_gpu/wf_reconstructor/modalwfreconstructorbuilder.cpp
    shs_on_gpu/wf_reconstructor/subclasses/wavefront.cpp
    shs_on_gpu/wf_reconstructor/subclasses/wfglocalresponsegenerator.cpp
    shs_on_gpu/wf_reconstructor/subclasses/wfgrad.cpp
    shs_on_gpu/wf_reconstructor/subclasses/wfgradmodegenerator.cpp
    shs_on_gpu/wf_reconstructor/subclasses/wfresponsesampler.cpp
    WFScamsim.c
    WFSmap.c
)

set(INCLUDEFILES
	${SRCNAME}.h
    AOloopControl_IOtools.h
    acquireWFSim.h
    acquireWFSspec.h
    ao188_preprocessor.h
    findspots.h
    shs_on_gpu/evaluator/SGE_CUDAkernel.hpp
    shs_on_gpu/evaluator/SGE_CUDAremoveTilt.hpp
    shs_on_gpu/evaluator/SGE_Evaluator.hpp
    shs_on_gpu/evaluator/SGE_Evaluator_interface.h
    shs_on_gpu/evaluator/SGE_GridLayout.hpp
    shs_on_gpu/evaluator/SGE_ReferenceManager.hpp
    shs_on_gpu/pupil_reshape.h
    shs_on_gpu/pupil_reshaper/SGE_Reshaper.hpp
    shs_on_gpu/pupil_reshaper/SGE_Reshaper_interface.h
    shs_on_gpu/pupil_zernike_gen.h
    shs_on_gpu/pupil_zern_generator/PupilZernikeGenerator_interface.h
    shs_on_gpu/pupil_zern_generator/PupilZernikeGenerator.hpp
    shs_on_gpu/ref_recorder/SGR_Recorder.hpp
    shs_on_gpu/ref_recorder/SGR_Recorder_interface.h
    shs_on_gpu/shs_gpu.h
    shs_on_gpu/shs_gpu_ref.h
    shs_on_gpu/util/atypeUtil.hpp
    shs_on_gpu/util/Circle.hpp
    shs_on_gpu/util/CudaUtil.hpp
    shs_on_gpu/util/GaussianKernel.hpp
    shs_on_gpu/util/ImageHandler2D.hpp
    shs_on_gpu/util/ImageHandler2DBase.hpp
    shs_on_gpu/util/OneDGaussianFit.hpp
    shs_on_gpu/util/Point.hpp
    shs_on_gpu/util/Pupil.hpp
    shs_on_gpu/util/Rectangle.hpp
    shs_on_gpu/util/SpotFitter.hpp
    shs_on_gpu/wf_reconstructor/modalwfreconstructor.hpp
    shs_on_gpu/wf_reconstructor/modalwfreconstructorbuilder.hpp
    shs_on_gpu/wf_reconstructor/subclasses/wavefront.hpp
    shs_on_gpu/wf_reconstructor/subclasses/wfglocalresponsegenerator.hpp
    shs_on_gpu/wf_reconstructor/subclasses/wfgrad.hpp
    shs_on_gpu/wf_reconstructor/subclasses/wfgradmodegenerator.hpp
    shs_on_gpu/wf_reconstructor/subclasses/wfresponsesampler.hpp
    WFScamsim.h
    WFSmap.h
)

set(SCRIPTS
    scripts/cacao-fpslistadd-LOG_STREAM
    scripts/cacao-fpslistadd-RESHAPE_1D_PUPIL
    scripts/cacao-fpslistadd-SHSONGPU_EVAL
    scripts/cacao-fpslistadd-SHSONGPU_REF
    scripts/cacao-setup-logging-fps
    scripts/specifications/cacao-fpslistadd-LOG_CMODES
    scripts/specifications/cacao-fpslistadd-LOG_SHSONGPU_SHS
    scripts/specifications/cacao-fpslistadd-LOG_SHSONGPU_GRAD
    scripts/specifications/cacao-fpslistadd-LOG_SHSONGPU_WF
    scripts/specifications/cacao-fpslistadd-LOG_SHSONGPU_INT
    scripts/specifications/cacao-fpslistadd-RSHP_ACQWF_AVG
    scripts/specifications/cacao-fpslistadd-RSHP_ACQWF_C
    scripts/specifications/cacao-fpslistadd-RSHP_SHSONGPU_GRAD
    scripts/specifications/cacao-fpslistadd-RSHP_SHSONGPU_INT
    scripts/specifications/cacao-fpslistadd-RSHP_SHSONGPU_WF
)

set(LINKLIBS
	CLIcore
    milkinfo
    milkCOREMODiofits
    milkZernikePolyn
    cacaoAOloopControl
    -lcublas
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
