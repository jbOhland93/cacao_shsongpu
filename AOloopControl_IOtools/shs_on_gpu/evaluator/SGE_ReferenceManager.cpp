#include "SGE_ReferenceManager.hpp"
#include "milkDebugTools.h"

#include <cuda.h>
#include <math.h>
#include "../util/CudaSGEutil.hpp"
#include "../ref_recorder/SGR_ReferenceKW.hpp"


SGE_ReferenceManager::SGE_ReferenceManager(
        IMAGE* ref,         // Stream holding the reference data
        IMAGE* cam,         // Stream holding the current SHS frame
        IMAGE* dark,        // Stream holding the dark frame of the SHS
        int deviceID)       // ID of the GPU device
    : m_deviceID(deviceID)
{
    printf("SGE_ReferenceManager Todo: Make factory functionto return shared ptr\n");

    checkInputStreamCoherence(ref, cam, dark);
    checkInputNamingCoherence(cam, dark);
    adoptReferenceStreamsFromKW();
    readShiftToGradConstantFromKW();
    generateGPUkernel();

    printf("SGE_ReferenceManager Todo: Build kernel and copy dark to GPU\n");
    // Todo: Build kernel and copy dark to GPU

    cudaError err;
    err = cudaSetDevice(m_deviceID);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceMapHost);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    printCE(err);

    copyDarkToGPU(dark);
}

SGE_ReferenceManager::~SGE_ReferenceManager()
{
    if (mdp_dark != nullptr)
        cudaFree(mdp_dark);
    if (mdp_kernel != nullptr)
        cudaFree(mdp_kernel);
}

void SGE_ReferenceManager::checkInputStreamCoherence(IMAGE* ref, IMAGE* cam, IMAGE* dark)
{
    // Check compatibility of camera and dark images
    int imWidth = cam->md->size[0];
    int imHeight = cam->md->size[1];
    if (imWidth != dark->md->size[0] || imHeight != dark->md->size[1])
        throw std::runtime_error("SGE_ReferenceManager: the camera and dark images are not of same size.\n");

    // Adopt the reference image
    if (!checkAtype<float>(ref->md->datatype))
        throw std::runtime_error("SGE_ReferenceManager: Reference has to be of type float.");
    mp_IHreference = SGR_ImageHandler<float>::newHandlerAdoptImage(ref->name);
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
            mp_IHmask = SGR_ImageHandler<uint8_t>::newHandlerAdoptImage(maskName.c_str());
            printf("\tAdopted the mask stream: %s\n", maskName.c_str());
        }
        
        std::string intensitySuffix;
        if (!mp_IHreference->getKeyword(REF_KW_INTENSITY_SUFFIX, &intensitySuffix))
            throw std::runtime_error("SGE_ReferenceManager: no intensity map suffix found in reference keywords. Could not retrieve the intensity stream name.\n");
        else
        {
            std::string intensityName = m_baseName;
            intensityName.append(intensitySuffix);
            mp_IHintensity = SGR_ImageHandler<float>::newHandlerAdoptImage(intensityName.c_str());
            printf("\tAdopted the intensity stream: %s\n", intensityName.c_str());
        }
    }
}

void SGE_ReferenceManager::readShiftToGradConstantFromKW()
{
    if (!mp_IHreference->getKeyword(REF_KW_SHIFT_2_GRAD_CONST, &m_shiftToGradConstant))
        throw std::runtime_error("SGE_ReferenceManager: shift-to-gradient constant not found in KWs, prohibiting WF reconstruction.");
}

void SGE_ReferenceManager::generateGPUkernel()
{
    printf("Generating correlation kernel ...\n");
    if (!mp_IHreference->getKeyword(REF_KW_KERNEL_SIZE, &m_kernelSize))
        throw std::runtime_error("SGE_ReferenceManager: kernel size not found in KWs. Cannot build correlation kernel.");
    if (!mp_IHreference->getKeyword(REF_KW_KERNEL_STDDEV, &m_kernelStdDev))
        throw std::runtime_error("SGE_ReferenceManager: kernel standard deviation not found in KWs. Cannot build correlation kernel.");
    printf("\tKernel properties found in KWs.\n");
    printf("Todo: Generate the kernel array and copy it to the GPU - store ptr in mdp_kernel.\n");
    printf("\tKernel generated in host memory.\n");
    printf("\tKernel copied to device.\n");
}

void SGE_ReferenceManager::copyDarkToGPU(IMAGE* imDark)
{
    printf("Copying darkframe to GPU ...\n");
    float* readBuf;
    ImageStreamIO_readLastWroteBuffer(imDark, (void**)&readBuf);

    cudaError_t err;
    cudaMalloc((void**)&mdp_dark, imDark->md->imdatamemsize);
    printCE(err);
    
    err = cudaMemcpy(mdp_dark, readBuf, imDark->md->imdatamemsize, cudaMemcpyHostToDevice);
    printCE(err);

    printf("\tDarkframe copied to device.\n");
}