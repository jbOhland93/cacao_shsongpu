#include "SGE_ReferenceManager.hpp"
#include "CLIcore.h"

#include <cuda.h>
#include <math.h>
#include "../ref_recorder/SGR_ReferenceKW.hpp"
#include "../util/Point.hpp"

spRefManager SGE_ReferenceManager::makeReferenceManager(
        IMAGE* cam,         // Camera stream
        IMAGE* dark,        // Dark stream
        IMAGE* refPos,              // Stream with SHS reference positions
        IMAGE* refMask,             // Stream with SHS reference mask
        IMAGE* refInt,              // Stream with SHS reference intensity
        std::string prefix) // Stream prefix
{
    return spRefManager(new SGE_ReferenceManager(cam, dark, refPos, refMask, refInt, prefix));
}

SGE_ReferenceManager::~SGE_ReferenceManager()
{
    if (mdp_dark != nullptr)
        cudaFree(mdp_dark);
    if (mdp_searchPosX != nullptr)
        cudaFree(mdp_searchPosX);
    if (mdp_searchPosY != nullptr)
        cudaFree(mdp_searchPosY);
    if (mdp_absRefX != nullptr)
        cudaFree(mdp_absRefX);
    if (mdp_absRefY != nullptr)
        cudaFree(mdp_absRefY);
}

void SGE_ReferenceManager::setUseAbsReference(bool useAbsoluteReference)
{
    m_useAbsRef = useAbsoluteReference;
}

float* SGE_ReferenceManager::getRefXGPU()
{
    if (m_useAbsRef)
        return mdp_absRefX;
    else
        return mp_IHreference->getGPUCopy();
}
float* SGE_ReferenceManager::getRefYGPU()
{
    if (m_useAbsRef)
        return mdp_absRefY;
    else
        return mp_IHreference->getGPUCopy() + m_numSpots;
}

SGE_ReferenceManager::SGE_ReferenceManager(
        IMAGE* cam,         // Stream holding the current SHS frame
        IMAGE* dark,        // Stream holding the dark frame of the SHS
        IMAGE* refPos,              // Stream with SHS reference positions
        IMAGE* refMask,             // Stream with SHS reference mask
        IMAGE* refInt,              // Stream with SHS reference intensity
        std::string prefix) // Stream prefix
    : m_streamPrefix(prefix)
{
    printf("\nSetting up reference manager ...\n");
    checkInputStreamCoherence(cam, dark, refPos, refMask, refInt);
    readConstantsFromKW();
    generateGPUkernel();
    copySearchPosToGPU();
    makeAbsRef();
    printf("Reference manager setup completed.\n");
}

void SGE_ReferenceManager::checkInputStreamCoherence(
    IMAGE* cam, IMAGE* dark, IMAGE* refPos, IMAGE* refMask, IMAGE* refInt)
{
    // Check compatibility of camera and dark images
    int imWidth = cam->md->size[0];
    int imHeight = cam->md->size[1];
    if (imWidth != dark->md->size[0] || imHeight != dark->md->size[1])
        throw std::runtime_error("SGE_ReferenceManager: the camera and dark images are not of same size.\n");
    if (!checkAtype<uint16_t>(cam->md->datatype))
        throw std::runtime_error("SGE_ReferenceManager: the camera stream has to be of type uint16_t.\n");
    if (!checkAtype<float>(dark->md->datatype))
        throw std::runtime_error("SGE_ReferenceManager: the dark stream has to be of type float.\n");

    // Adopt the reference image
    if (!checkAtype<float>(refPos->md->datatype))
        throw std::runtime_error("SGE_ReferenceManager: reference has to be of type float.\n");
    mp_IHreference = ImageHandler2D<float>::newHandler2DAdoptImage(refPos->name);
    m_numSpots = mp_IHreference->mWidth;

    // Check reference positions against camera image size
    if (mp_IHreference->mHeight != 2)
        throw std::runtime_error("SGE_ReferenceManager: The reference does not feature a height of 2.");
    float* refData = mp_IHreference->getWriteBuffer();
    for (size_t i = 0; i < m_numSpots; i++)
    {
        float refX = refData[i];
        float refY = refData[i+mp_IHreference->mWidth];
        if (refX < 0 || refX >= imWidth || refY < 0 || refY >= imHeight)
            throw std::runtime_error("SGE_ReferenceManager: Reference positions outside of image bounds.");
    }

    // Adopt the mask image
    if (!checkAtype<uint8_t>(refMask->md->datatype))
        throw std::runtime_error("SGE_ReferenceManager: reference mask has to be of type uint8_t or float.\n");
    mp_IHmask = ImageHandler2D<uint8_t>::newHandler2DAdoptImage(refMask->name);
    // Verify number of spots
    int spotCounter = 0;
    for (size_t ix = 0; ix < mp_IHmask->mWidth; ix++)
        for (size_t iy = 0; iy < mp_IHmask->mHeight; iy++)
            if ( mp_IHmask->read(ix, iy) != 0)
                spotCounter ++;
    if (spotCounter != m_numSpots)
        throw std::runtime_error("SGE_ReferenceManager: number of valid mask samples does not equal reference size.\n");

    // Adopt the intensity image
    if (!checkAtype<float>(refInt->md->datatype))
        throw std::runtime_error("SGE_ReferenceManager: reference intensity has to be of type float.\n");
    mp_IHintensity = ImageHandler2D<float>::newHandler2DAdoptImage(refInt->name);
    if (mp_IHintensity->mWidth != mp_IHmask->mWidth ||
        mp_IHintensity->mHeight != mp_IHmask->mHeight)
        throw std::runtime_error("SGE_ReferenceManager: reference intensity has to feature the same size as the mask.\n");
}

void SGE_ReferenceManager::readConstantsFromKW()
{
    if (!mp_IHreference->getKeyword(REF_KW_PX_PITCH, &m_pixelPitch))
        throw std::runtime_error("SGE_ReferenceManager: pixel pitch not found in KWs, prohibiting absolute reference calculation.");

    if (!mp_IHreference->getKeyword(REF_KW_SHIFT_2_GRAD_CONST, &m_shiftToGradConstant))
        throw std::runtime_error("SGE_ReferenceManager: shift-to-gradient constant not found in KWs, prohibiting WF reconstruction.");
}

void SGE_ReferenceManager::generateGPUkernel()
{
    printf("Generating correlation kernel ...\n");
    double kernelStdDev;
    if (!mp_IHreference->getKeyword(REF_KW_KERNEL_STDDEV, &kernelStdDev))
        throw std::runtime_error("SGE_ReferenceManager: kernel size not found in KWs. Cannot build correlation kernel.");
    printf("\tKernel properties found in KWs.\n\t");
    std::string kernelStreamname = m_streamPrefix.append("kernel");
    mp_kernel = GaussianKernel::makeKernel((float) kernelStdDev, kernelStreamname.c_str(), false);
    printf("\tKernel generated in host memory.\n");
}

void SGE_ReferenceManager::copySearchPosToGPU()
{
    // Get the initial search positions
    // This is the reference positions, rounded to nearest
    uint16_t searchPosX[m_numSpots];
    uint16_t searchPosY[m_numSpots];
    float* ref = mp_IHreference->getWriteBuffer();
    for (uint16_t i = 0; i < m_numSpots; i++)
    {
        searchPosX[i] = (uint16_t) round(ref[i]);
        searchPosY[i] = (uint16_t) round(ref[i+m_numSpots]); 
    }
    // Allocate device memory and copy data to device
    unsigned long bufsize = sizeof(uint16_t)*m_numSpots;
    cudaMalloc((void**)&mdp_searchPosX, bufsize);
    cudaMalloc((void**)&mdp_searchPosY, bufsize);
    cudaMemcpy(mdp_searchPosX, searchPosX, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpy(mdp_searchPosY, searchPosY, bufsize, cudaMemcpyHostToDevice);
}

void SGE_ReferenceManager::makeAbsRef()
{
    printf("Generating absolut ereference in device memory ...\n");
    Point<double> pitch(m_pixelPitch, m_pixelPitch);
    Point<double> halfPitch = pitch / 2;
    float* refPtr = mp_IHreference->getWriteBuffer();
    Point<double> gridAnchor(refPtr[0], refPtr[m_numSpots]);
    gridAnchor -= halfPitch;

    printf("\tFinding grid anchor point ...\n");
    int numIterations = 5;
    for (int n = 0; n < numIterations; n++)
    {
        Point<double> distanceToNext(0, 0);
        Point<double> globalDisplacement(0, 0);
        double currentError = 0;
        for (int i = 0; i < m_numSpots; i++)
        {
            Point<float>spot(refPtr[i], refPtr[i+m_numSpots]);
            distanceToNext = (spot - gridAnchor) % pitch - halfPitch;
            globalDisplacement += distanceToNext;
            currentError += distanceToNext.abs()*distanceToNext.abs();
        }
        currentError = sqrt(currentError/m_numSpots);
        globalDisplacement /= m_numSpots;
        gridAnchor += globalDisplacement;
        printf("\t\tIteration %d, position error (RMS in px): %.6f\n", n, currentError);
    }
    
    printf("\tSpanning reference grid ...\n");
    float absRefX[m_numSpots];
    float absRefY[m_numSpots];
    Point<double> distanceToNext(0, 0);
    for (int i = 0; i < m_numSpots; i++)
    {
        Point<float>spot(refPtr[i], refPtr[i+m_numSpots]);
        distanceToNext = (spot - gridAnchor) % pitch - halfPitch;
        Point<float>absolutPos = spot + distanceToNext;
        absRefX[i] = absolutPos.mX;
        absRefY[i] = absolutPos.mY;
    }

    printf("\tCopy absolute reference to buffer ...\n");
    // Allocate device memory
    unsigned long bufsize = sizeof(float)*m_numSpots;
    if (mdp_absRefX == nullptr)
        cudaMalloc((void**)&mdp_absRefX, bufsize);
    if (mdp_absRefY == nullptr)
        cudaMalloc((void**)&mdp_absRefY, bufsize);

    cudaMemcpy(mdp_absRefX, absRefX, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpy(mdp_absRefY, absRefY, bufsize, cudaMemcpyHostToDevice);
}