#ifndef SGE_REFERENCEMANAGER_HPP
#define SGE_REFERENCEMANAGER_HPP

#include "../ref_recorder/SGR_ImageHandler.hpp"
#include <errno.h>
#include "../util/GaussianKernel.hpp"

#define spRefManager std::shared_ptr<SGE_ReferenceManager>

// A class for evaluating SHS images on a GPU
class SGE_ReferenceManager
{
public:
    // Factory function
    static spRefManager makeReferenceManager(
        IMAGE* ref,         // Stream holding the reference data
        IMAGE* cam,         // Camera stream
        IMAGE* dark,        // Dark stream
        std::string prefix);// Stream prefix
    // Dtor
    ~SGE_ReferenceManager();

    uint16_t getNumSpots() { return m_numSpots; }
    int64_t getKernelSize() { return mp_kernel->getKernelSize(); }
    float* getGPUdarkBuffer() { return mdp_dark; }
    float* getGPUkernelBuffer() { return mp_kernel->getPointerToDeviceCopy(); }

private:
    std::string m_streamPrefix;
    uint16_t m_numSpots;
    double m_shiftToGradConstant;
    spGKernel mp_kernel;

    // Reference images, adopted
    spImageHandler(float) mp_IHreference = nullptr;
    spImageHandler(float) mp_IHmask = nullptr; // Change back to uint8_t once fits writing is fixed
    spImageHandler(float) mp_IHintensity = nullptr;
    // Image arrays on device
    float* mdp_dark = nullptr;
    // Base name of the reference
    std::string m_baseName;

    SGE_ReferenceManager(); // No publically available Ctor
    // Ctor, doing the initialization
    SGE_ReferenceManager(
        IMAGE* ref,         // Stream holding the reference data
        IMAGE* cam,         // Camera stream
        IMAGE* dark,        // Dark stream
        std::string prefix);// Stream prefix

    // Helper functions
    void checkInputStreamCoherence(IMAGE* ref, IMAGE* cam, IMAGE* dark);
    void checkInputNamingCoherence(IMAGE* cam, IMAGE* dark);
    void adoptReferenceStreamsFromKW();
    void readShiftToGradConstantFromKW();
    void generateGPUkernel();
    void copyDarkToGPU(IMAGE* imDark);
};

#endif // SGE_REFERENCEMANAGER_HPP
