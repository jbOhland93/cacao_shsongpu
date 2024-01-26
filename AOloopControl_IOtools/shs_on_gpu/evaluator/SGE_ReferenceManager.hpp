#ifndef SGE_REFERENCEMANAGER_HPP
#define SGE_REFERENCEMANAGER_HPP

#include "../util/ImageHandler2D.hpp"
#include "../util/GaussianKernel.hpp"
#include <errno.h>

#define spRefManager std::shared_ptr<SGE_ReferenceManager>

// A class for evaluating SHS images on a GPU
class SGE_ReferenceManager
{
public:
    // Factory function
    static spRefManager makeReferenceManager(
        IMAGE* cam,         // Camera stream
        IMAGE* dark,        // Dark stream
        IMAGE* refPos,              // Stream with SHS reference positions
        IMAGE* refMask,             // Stream with SHS reference mask
        IMAGE* refInt,              // Stream with SHS reference intensity
        std::string prefix);// Stream prefix
    // Dtor
    ~SGE_ReferenceManager();

    // If true, a regular grid will be fitted to the reference,
    // determining the theoretical ideal spot positions.
    // Can be reversed by calling this function with false.
    void setUseAbsReference(bool useAbsoluteReference);

    spImHandler2D(float) getRefIH() { return mp_IHreference; }
    spImHandler2D(uint8_t) getMaskIH() { return mp_IHmask; }
    spImHandler2D(float) getIntensityIH() { return mp_IHintensity; }

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
    spImHandler2D(float) mp_IHreference = nullptr;
    spImHandler2D(uint8_t) mp_IHmask = nullptr;
    spImHandler2D(float) mp_IHintensity = nullptr;
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
        IMAGE* cam,         // Camera stream
        IMAGE* dark,        // Dark stream
        IMAGE* refPos,              // Stream with SHS reference positions
        IMAGE* refMask,             // Stream with SHS reference mask
        IMAGE* refInt,              // Stream with SHS reference intensity
        std::string prefix);// Stream prefix

    // Helper functions
    void checkInputStreamCoherence(IMAGE* cam, IMAGE* refPos, IMAGE* refMask, IMAGE* refInt, IMAGE* dark);
    void readConstantsFromKW();
    void generateGPUkernel();
    void copySearchPosToGPU();
    void makeAbsRef();
};

#endif // SGE_REFERENCEMANAGER_HPP
