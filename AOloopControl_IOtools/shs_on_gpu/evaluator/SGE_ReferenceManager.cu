#include "SGE_ReferenceManager.hpp"
#include "CLIcore.h"

#include <cuda.h>
#include <math.h>
#include "../ref_recorder/SGR_ReferenceKW.hpp"
#include "../util/Point.hpp"

spRefManager SGE_ReferenceManager::makeReferenceManager(
        IMAGE* ref,         // Stream holding the reference data
        IMAGE* cam,         // Camera stream
        IMAGE* dark,        // Dark stream
        std::string prefix) // Stream prefix
{
    return spRefManager(new SGE_ReferenceManager(ref, cam, dark, prefix));
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
        IMAGE* ref,         // Stream holding the reference data
        IMAGE* cam,         // Stream holding the current SHS frame
        IMAGE* dark,        // Stream holding the dark frame of the SHS
        std::string prefix) // Stream prefix
    : m_streamPrefix(prefix)
{
    printf("\nSetting up reference manager ...\n");
    checkInputStreamCoherence(ref, cam, dark);
    checkInputNamingCoherence(cam, dark);
    adoptReferenceStreamsFromKW();
    readConstantsFromKW();
    generateGPUkernel();
    copySearchPosToGPU();
    makeAbsRef();
    printf("Reference manager setup completed.\n");
}

void SGE_ReferenceManager::checkInputStreamCoherence(IMAGE* ref, IMAGE* cam, IMAGE* dark)
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
    if (!checkAtype<float>(ref->md->datatype))
        throw std::runtime_error("SGE_ReferenceManager: reference has to be of type float.\n");
    mp_IHreference = ImageHandler<float>::newHandlerAdoptImage(ref->name);
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
}

void SGE_ReferenceManager::checkInputNamingCoherence(IMAGE* cam, IMAGE* dark)
{
    printf("Checking camera and dark stream names against names stored in reference KWs ...\n");
    std::string inputName;
    if (mp_IHreference->getKeyword(REF_KW_INPUT_NAME, &inputName))
    {
        std::string compareName = std::string(cam->name);
        if (compareName.length() > 16)
            compareName = compareName.substr(0, 16);
        if (compareName == inputName)
            printf("\tThe shs camera stream name matches the name stored in the reference.\n");
        else
        {
            printf("WARNING: shs camera stream name does not match the name stored in the reference!\n");
            printf("\tExpected name (trimmed to 16 characters): %s\n", inputName.c_str());
            printf("\tActual name (trimmed to 16 characters): %s\n", compareName.c_str());
        }
    }
    else
        printf("WARNING: no shs camera stream name found in reference keywords! Unable to check integrity.\n");

    std::string darkName;
    if (!mp_IHreference->getKeyword(REF_KW_DARK_NAME, &darkName))
        printf("WARNING: no shs dark stream name found in reference keywords! Unable to check integrity.\n");
    else
    {
        std::string compareName = std::string(dark->name);
        if (compareName.length() > 16)
            compareName = compareName.substr(0, 16);
        if (compareName == darkName)
            printf("\tThe shs dark stream name matches the name stored in the reference.\n");
        else
        {
            printf("WARNING: shs dark stream name does not match the name stored in the reference!\n");
            printf("\tExpected name (trimmed to 16 characters): %s\n", darkName.c_str());
            printf("\tActual name (trimmed to 16 characters): %s\n", compareName.c_str());
        }
    }
}

void SGE_ReferenceManager::adoptReferenceStreamsFromKW()
{
    // Extract info from the reference image keywords
    printf("Parsing reference keywords, adopting mask and intensity reference streams...\n");
    
    int64_t suffixLength;
    if (!mp_IHreference->getKeyword(REF_KW_SUFFIX_LEN, &suffixLength))
        throw std::runtime_error("SGE_ReferenceManager: no suffix length found in reference keywords. Cannot retrieve the stream names of the mask and intensity map.");
    else
    {
        m_baseName = mp_IHreference->getImage()->name;
        m_baseName = m_baseName.substr(0, m_baseName.length()-suffixLength);
        std::string maskSuffix;
        if (!mp_IHreference->getKeyword(REF_KW_MASK_SUFFIX, &maskSuffix))
            throw std::runtime_error("SGE_ReferenceManager: no mask suffix found in reference keywords. Could not retrieve the mask stream name.\n");
        else
        {
            std::string maskName = m_baseName;
            maskName.append(maskSuffix);
            mp_IHmask = ImageHandler<uint8_t>::newHandlerAdoptImage(maskName.c_str());
            printf("\tAdopted the mask stream: %s\n", maskName.c_str());
        }
        
        std::string intensitySuffix;
        if (!mp_IHreference->getKeyword(REF_KW_INTENSITY_SUFFIX, &intensitySuffix))
            throw std::runtime_error("SGE_ReferenceManager: no intensity map suffix found in reference keywords. Could not retrieve the intensity stream name.\n");
        else
        {
            std::string intensityName = m_baseName;
            intensityName.append(intensitySuffix);
            mp_IHintensity = ImageHandler<float>::newHandlerAdoptImage(intensityName.c_str());
            printf("\tAdopted the intensity stream: %s\n", intensityName.c_str());
        }
    }
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