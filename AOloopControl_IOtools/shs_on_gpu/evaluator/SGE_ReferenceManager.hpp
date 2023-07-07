#ifndef SGE_REFERENCEMANAGER_HPP
#define SGE_REFERENCEMANAGER_HPP

#include "../ref_recorder/SGR_ImageHandler.hpp"
#include <errno.h>

// A class for evaluating SHS images on a GPU
class SGE_ReferenceManager
{
public:
    // Ctor, doing the initialization
    SGE_ReferenceManager(
        IMAGE* ref,         // Stream holding the reference data
        IMAGE* cam,         // Camera stream
        IMAGE* dark,        // Dark stream
        int deviceID = 0);  // ID of the GPU device
    // Dtor
    ~SGE_ReferenceManager();

    int getDeviceID() { return m_deviceID; }
    uint16_t getNumSpots() { return m_numSpots; }
    int64_t getKernelSize() { return m_kernelSize; }
    float* getGPUdarkBuffer() { return mdp_dark; }
    float* getGPUkernelBuffer() { return mdp_kernel; }

private:
    int m_deviceID;
    uint16_t m_numSpots;
    double m_kernelStdDev;
    int64_t m_kernelSize;
    double m_shiftToGradConstant;

    // Reference images, adopted
    spImageHandler(float) mp_IHreference = nullptr;
    spImageHandler(uint8_t) mp_IHmask = nullptr;
    spImageHandler(float) mp_IHintensity = nullptr;
    // Image arrays on device
    float* mdp_dark = nullptr;
    float* mdp_kernel = nullptr;
    // Base name of the reference
    std::string m_baseName;

    // Helper functions
    void checkInputStreamCoherence(IMAGE* ref, IMAGE* cam, IMAGE* dark);
    void checkInputNamingCoherence(IMAGE* cam, IMAGE* dark);
    void adoptReferenceStreamsFromKW();
    void readShiftToGradConstantFromKW();
    void generateGPUkernel();
    void copyDarkToGPU(IMAGE* imDark);
};

#endif // SGE_REFERENCEMANAGER_HPP
