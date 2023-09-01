#include "ImageHandlerBase.hpp"

#include <cuda.h>
#include "CudaUtil.hpp"



ImageHandlerBase::~ImageHandlerBase()
{   
    // Destroy the image only if persistent is not enabled.
    if (!mPersistent)
        ImageStreamIO_destroyIm(mpImage);

    // Clean up device memory in any case
    if (mpd_dataGPU != nullptr)
        cudaFree(mpd_dataGPU);
    
    delete mpImage;
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
        uint32_t height)
        :
        mWidth(width),
        mHeight(height),
        mNumPx(width*height),
        mROI(0,0,width,height)
{
    mpImage = new IMAGE();
}

void* ImageHandlerBase::getDeviceCopy()
{
    if (m_gpuCopySize != mpImage->md->imdatamemsize)
        updateDeviceCopy();
    return mpd_dataGPU;
}

void ImageHandlerBase::updateDeviceCopy()
{
    cudaError_t err;
    if (m_gpuCopySize != mpImage->md->imdatamemsize && mpd_dataGPU != nullptr) 
    {
            err = cudaFree(mpd_dataGPU);
            printCE(err);
            mpd_dataGPU = nullptr;
    }
    if (mpd_dataGPU == nullptr)
    {
        m_gpuCopySize = mpImage->md->imdatamemsize;
        err = cudaMalloc((void**)&mpd_dataGPU, m_gpuCopySize);
        printCE(err);
    }
    // Only perform the copy if the host image has been updated
    if (mpImage->md->cnt0 != mCnt0deviceCopy)
    {
        void* src;
        ImageStreamIO_readLastWroteBuffer(mpImage, &src);
        err = cudaMemcpy(mpd_dataGPU, src, m_gpuCopySize, cudaMemcpyHostToDevice);
        printCE(err);
        mCnt0deviceCopy = mpImage->md->cnt0;
    }
}

void ImageHandlerBase::updateFromDevice()
{
    if (mpd_dataGPU == nullptr)
        throw std::runtime_error("ImageHandlerBase::updateFromDevice: No device copy used.\n");
    if (m_gpuCopySize != mpImage->md->imdatamemsize)
        throw std::runtime_error("ImageHandlerBase::updateFromDevice: Array size mismatch.\n");
    
    void* dst;
    ImageStreamIO_readLastWroteBuffer(mpImage, &dst);
    cudaError_t err;
    err = cudaMemcpy(dst, mpd_dataGPU, m_gpuCopySize, cudaMemcpyDeviceToHost);
    printCE(err);
    ImageStreamIO_UpdateIm(mpImage);
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

int ImageHandlerBase::getKWindex(std::string name)
{
    for (int i = 0; i < mpImage->md->NBkw; i++)
    {
        std::string kwName = mpImage->kw[i].name;
        while(kwName.length() > name.length())
            kwName.pop_back();
        if (name == kwName)
            return i;
    }
    return -1;
}