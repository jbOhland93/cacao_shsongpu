#ifndef SGE_REFERENCEMANAGER_HPP
#define SGE_REFERENCEMANAGER_HPP

#include "../util/ImageHandler.hpp"
#include "../util/GaussianKernel.hpp"
#include <errno.h>

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

    // If true, a regular grid will be fitted to the reference,
    // determining the theoretical ideal spot positions.
    // Can be reversed by calling this function with false.
    void setUseAbsReference(bool useAbsoluteReference);

    spImageHandler(float) getRefIH() { return mp_IHreference; }
    spImageHandler(uint8_t) getMaskIH() { return mp_IHmask; }
    spImageHandler(float) getIntensityIH() { return mp_IHintensity; }

    uint16_t getNumSpots() { return m_numSpots; }
    int64_t getKernelSize() { return mp_kernel->getKernelSize(); }
    float* getKernelBufferGPU() { return mp_kernel->getPointerToDeviceCopy(); }
    uint16_t* getSearchPosXGPU() { return mdp_searchPosX; }
    uint16_t* getSearchPosYGPU() { return mdp_searchPosY; }
    float* getRefXGPU();
    float* getRefYGPU();
    float getShiftToGradConstant() { return (float) m_shiftToGradConstant; }
    double getPixelPitch() { return m_pixelPitch; }

private:
    std::string m_streamPrefix;
    uint16_t m_numSpots;
    bool m_useAbsRef = false;
    double m_pixelPitch;
    double m_shiftToGradConstant;
    spGKernel mp_kernel;

    // Reference images, adopted
    spImageHandler(float) mp_IHreference = nullptr;
    spImageHandler(uint8_t) mp_IHmask = nullptr;
    spImageHandler(float) mp_IHintensity = nullptr;
    // Image arrays on device
    float* mdp_dark = nullptr;
    uint16_t* mdp_searchPosX = nullptr;
    uint16_t* mdp_searchPosY = nullptr;
    float* mdp_absRefX = nullptr;
    float* mdp_absRefY = nullptr;
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
    void readConstantsFromKW();
    void generateGPUkernel();
    void copySearchPosToGPU();
    void makeAbsRef();
};

#endif // SGE_REFERENCEMANAGER_HPP
