#include "ImageHandler2DBase.hpp"

#include <cuda.h>
#include "CudaUtil.hpp"

ImageHandler2DBase::~ImageHandler2DBase()
{   
    // Destroy the image only if persistent is not enabled.
    if (!mPersistent)
        ImageStreamIO_destroyIm(mp_image);

    // Clean up device memory in any case
    if (mp_d_imData != nullptr)
        cudaFree(mp_d_imData);
    
    delete mp_image;
}

cudaError_t ImageHandler2DBase::mapImForGPUaccess()
{
    return cudaHostRegister(
            ImageStreamIO_get_image_d_ptr(mp_image),
            mp_image->md->imdatamemsize,
            cudaHostRegisterMapped);
}

void ImageHandler2DBase::setSlice(uint32_t sliceIndex)
{
    if (sliceIndex >= mDepth)
        throw std::runtime_error("SGR_ImageHandler::setSlice: out of range.");
    else
        m_currentSlice = sliceIndex;
}

void ImageHandler2DBase::setROI(Rectangle<uint32_t> roi)
{
    if (roi.x()+roi.w() >= mWidth || roi.y()+roi.h() >= mHeight)
        throw std::runtime_error("SGR_ImageHandler::setROI: out of range.");
    else
        mROI = roi;
}

void ImageHandler2DBase::setROI(uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    setROI(Rectangle<uint32_t>(x,y,w,h));
}

void ImageHandler2DBase::unsetROI()
{
    mROI = Rectangle<uint32_t>(0,0, mWidth, mHeight);
}

bool ImageHandler2DBase::waitForNextFrame(int timeout_us)
{
    timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    // Add timeout to timespec - and ansure tv_nsec is within its limits
    timeout.tv_nsec += (__time_t) timeout_us*1000;
    timeout.tv_sec += timeout.tv_nsec / (long) 1e9;
    timeout.tv_nsec = timeout.tv_nsec % (long) 1e9;

    uint64_t lastFrameIndex = mp_image->md->cnt0;
    ImageStreamIO_semtimedwait(mp_image, m_semaphoreIndex, &timeout);
    if (mp_image->md->cnt0 != lastFrameIndex)
    {
        ImageStreamIO_semflush(mp_image, m_semaphoreIndex);
        return true;
    }
    else
        return false;
}

void ImageHandler2DBase::saveToFPSdataDir(FUNCTION_PARAMETER_STRUCT* fps, std::string fname)
{
    read_sharedmem_image(mp_image->name);
    fps_write_RUNoutput_image(fps, mp_image->name, fname.c_str());
}

ImageHandler2DBase::ImageHandler2DBase(
        uint32_t width,
        uint32_t height,
        uint32_t depth)
        :
        mWidth(width),
        mHeight(height),
        mDepth(depth),
        mNumPx(width*height*depth),
        mROI(0,0,width,height)
{
    mp_image = new IMAGE();
}

void ImageHandler2DBase::updateImMetadata()
{
    mp_h_imData = ImageStreamIO_get_image_d_ptr(mp_image);
    m_dataSize = mp_image->md->imdatamemsize;
    m_semaphoreIndex =
            ImageStreamIO_getsemwaitindex(mp_image, m_semaphoreIndex);
}

void* ImageHandler2DBase::getDeviceCopy()
{
    if (m_gpuCopySize != mp_image->md->imdatamemsize)
        updateDeviceCopy();
    return mp_d_imData;
}

void ImageHandler2DBase::updateDeviceCopy()
{
    cudaError_t err;
    if (m_gpuCopySize != mp_image->md->imdatamemsize && mp_d_imData != nullptr) 
    {
            err = cudaFree(mp_d_imData);
            printCE(err);
            mp_d_imData = nullptr;
    }
    if (mp_d_imData == nullptr)
    {
        m_gpuCopySize = mp_image->md->imdatamemsize;
        err = cudaMalloc((void**)&mp_d_imData, m_gpuCopySize);
        printCE(err);
    }
    // Only perform the copy if the host image has been updated
    if (mp_image->md->cnt0 != mCnt0deviceCopy)
    {
        err = cudaMemcpy(
            mp_d_imData,
            mp_h_imData,
            m_gpuCopySize,
            cudaMemcpyHostToDevice);
        printCE(err);
        mCnt0deviceCopy = mp_image->md->cnt0;
    }
}

void ImageHandler2DBase::updateFromDevice()
{
    if (mp_d_imData == nullptr)
        throw std::runtime_error("ImageHandler2DBase::updateFromDevice: No device copy used.\n");
    if (m_gpuCopySize != mp_image->md->imdatamemsize)
        throw std::runtime_error("ImageHandler2DBase::updateFromDevice: Array size mismatch.\n");
    
    void* dst;
    ImageStreamIO_readLastWroteBuffer(mp_image, &dst);
    cudaError_t err;
    err = cudaMemcpy(
        dst,
        mp_d_imData,
        m_gpuCopySize,
        cudaMemcpyDeviceToHost);
    printCE(err);
    ImageStreamIO_UpdateIm(mp_image);
}

uint32_t ImageHandler2DBase::fromROIxToImX(uint32_t x)
{
    if (x >= mROI.w()) // x is uint3_t, thus always > 0
        throw std::runtime_error("ImageHandler2DBase::toROIx: x is out of range.");
    else
        return x + mROI.x();
}

uint32_t ImageHandler2DBase::fromROIyToImY(uint32_t y)
{
    if (y >= mROI.h()) // y is uint3_t, thus always > 0
        throw std::runtime_error("ImageHandler2DBase::toROIy: y is out of range.");
    else
        return y + mROI.y();
}

std::vector<uint32_t> ImageHandler2DBase::getSizeVector()
{
    return std::vector<uint32_t>({mWidth, mHeight, mDepth});
}

int ImageHandler2DBase::getKWindex(std::string name)
{
    for (int i = 0; i < mp_image->md->NBkw; i++)
    {
        std::string kwName = mp_image->kw[i].name;
        while(kwName.length() > name.length())
            kwName.pop_back();
        if (name == kwName)
            return i;
    }
    return -1;
}