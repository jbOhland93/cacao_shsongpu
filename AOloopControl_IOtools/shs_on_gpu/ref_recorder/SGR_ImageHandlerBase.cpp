#include "SGR_ImageHandlerBase.hpp"

SGR_ImageHandlerBase::~SGR_ImageHandlerBase()
{
    if (!mPersistent)
        ImageStreamIO_destroyIm(&mImage);
}

void SGR_ImageHandlerBase::setROI(Rectangle<uint32_t> roi)
{
    if (roi.x()+roi.w() >= mWidth || roi.y()+roi.h() >= mHeight)
            throw std::runtime_error("SGR_ImageHandler::setROI: out of range.");
        else
            mROI = roi;
}

void SGR_ImageHandlerBase::setROI(uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    setROI(Rectangle<uint32_t>(x,y,w,h));
}

void SGR_ImageHandlerBase::unsetROI()
{
    mROI = Rectangle<uint32_t>(0,0, mWidth, mHeight);
}

SGR_ImageHandlerBase::SGR_ImageHandlerBase(
        uint32_t width,
        uint32_t height,
        int32_t gpuDevice)
        :
        mWidth(width),
        mHeight(height),
        mNumPx(width*height),
        mROI(0,0,width,height),
        mDevice(gpuDevice)
{}

uint32_t SGR_ImageHandlerBase::fromROIxToImX(uint32_t x)
{
    if (x >= mROI.w()) // x is uint3_t, thus always > 0
        throw std::runtime_error("SGR_ImageHandlerBase::toROIx: x is out of range.");
    else
        return x + mROI.x();
}

uint32_t SGR_ImageHandlerBase::fromROIyToImY(uint32_t y)
{
    if (y >= mROI.h()) // y is uint3_t, thus always > 0
        throw std::runtime_error("SGR_ImageHandlerBase::toROIy: y is out of range.");
    else
        return y + mROI.y();
}

int SGR_ImageHandlerBase::getKWindex(std::string name)
{
    for (int i = 0; i < mImage.md->NBkw; i++)
    {
        std::string kwName = mImage.kw[i].name;
        while(kwName.length() > name.length())
            kwName.pop_back();
        if (name == kwName)
            return i;
    }
    return -1;
}