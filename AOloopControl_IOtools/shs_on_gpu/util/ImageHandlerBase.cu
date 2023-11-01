#include "ImageHandlerBase.hpp"

#include <cuda.h>
#include "CudaUtil.hpp"

ImageHandlerBase::~ImageHandlerBase()
{   
    // Destroy the image only if persistent is not enabled.
    if (!mPersistent)
        ImageStreamIO_destroyIm(mp_image);

    // Clean up device memory in any case
    if (mp_d_imData != nullptr)
        cudaFree(mp_d_imData);
    
    delete mp_image;
}

cudaError_t ImageHandlerBase::mapImForGPUaccess()
{
    return cudaHostRegister(
            ImageStreamIO_get_image_d_ptr(mp_image),
            mp_image->md->imdatamemsize,
            cudaHostRegisterMapped);
}

void ImageHandlerBase::setSlice(uint32_t sliceIndex)
{
    if (sliceIndex >= mDepth)
        throw std::runtime_error("SGR_ImageHandler::setSlice: out of range.");
    else
        m_currentSlice = sliceIndex;
}

void ImageHandlerBase::setROI(Rectangle<uint32_t> roi)
{
    if (roi.x()+roi.w() >= mWidth || roi.y()+roi.h() >= mHeight)
        throw std::runtime_error("SGR_ImageHandler::setROI: out of range.");
    else
        mROI = roi;
}

void ImageHandlerBase::setROI(uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    setROI(Rectangle<uint32_t>(x,y,w,h));
}

void ImageHandlerBase::unsetROI()
{
    mROI = Rectangle<uint32_t>(0,0, mWidth, mHeight);
}

ImageHandlerBase::ImageHandlerBase(
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

void ImageHandlerBase::updateImMetadata()
{
    mp_h_imData = ImageStreamIO_get_image_d_ptr(mp_image);
    m_dataSize = mp_image->md->imdatamemsize;
}

void* ImageHandlerBase::getDeviceCopy()
{
    if (m_gpuCopySize != mp_image->md->imdatamemsize)
        updateDeviceCopy();
    return mp_d_imData;
}

void ImageHandlerBase::updateDeviceCopy()
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

void ImageHandlerBase::updateFromDevice()
{
    if (mp_d_imData == nullptr)
        throw std::runtime_error("ImageHandlerBase::updateFromDevice: No device copy used.\n");
    if (m_gpuCopySize != mp_image->md->imdatamemsize)
        throw std::runtime_error("ImageHandlerBase::updateFromDevice: Array size mismatch.\n");
    
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

uint32_t ImageHandlerBase::fromROIxToImX(uint32_t x)
{
    if (x >= mROI.w()) // x is uint3_t, thus always > 0
        throw std::runtime_error("ImageHandlerBase::toROIx: x is out of range.");
    else
        return x + mROI.x();
}

uint32_t ImageHandlerBase::fromROIyToImY(uint32_t y)
{
    if (y >= mROI.h()) // y is uint3_t, thus always > 0
        throw std::runtime_error("ImageHandlerBase::toROIy: y is out of range.");
    else
        return y + mROI.y();
}

std::vector<uint32_t> ImageHandlerBase::getSizeVector()
{
    return std::vector<uint32_t>({mWidth, mHeight, mDepth});
}

int ImageHandlerBase::getKWindex(std::string name)
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