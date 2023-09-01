// This class provides basic functionality for the more specific SGR_ImageHandler

#ifndef IMAGEHANDLERBASE_HPP
#define IMAGEHANDLERBASE_HPP

#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "ImageStreamIO/ImageStreamIO.h"
#include "../util/Rectangle.hpp"

class ImageHandlerBase
{
public:
    // The width of the image
    const uint32_t mWidth;
    // The height of the image
    const uint32_t mHeight;
    // The total number of pixels in the image
    const uint32_t mNumPx;

    ~ImageHandlerBase();

    // Returns the image object
    IMAGE* getImage() { return mpImage; }
    // Returns the image array size in memory
    size_t getBufferSize() { return mpImage->md->imdatamemsize; }

    // Makes the image stream stay after destruction
    void setPersistent(bool persistent) { mPersistent = persistent; }

    // Setting a ROI
    void setROI(Rectangle<uint32_t> roi);
    void setROI(uint32_t x, uint32_t y, uint32_t w, uint32_t h);
    // Resetting the ROI
    void unsetROI();
    // Geting the ROI
    Rectangle<uint32_t> getROI() { return mROI; }
    // Get the number of pixels in the current ROI
    uint32_t getPxInROI() { return mROI.w() * mROI.h(); }

    // Setting a keyword
    template <typename U>
    void setKeyword(int index, std::string name, U data);
    // Reading a keyword
    template <typename U>
    bool getKeyword(std::string name, U* dst);

    // Updates the image
    void updateWrittenImage() { ImageStreamIO_UpdateIm(mpImage); }
    
protected:
    // The image, managed by this class
    IMAGE* mpImage;
    // If false, the image will be destoyed with the desturction of this instance
    bool mPersistent = false;
    // The cnt0 value of the last frame that has been copied to the GPU.
    uint64_t mCnt0deviceCopy = std::numeric_limits<uint64_t>::max();
    // The region of interest. May be used by processing routines.
    Rectangle<uint32_t> mROI = Rectangle<uint32_t>(0, 0, 0, 0);

    // Ctor
    ImageHandlerBase(uint32_t width, uint32_t height);

    // Returns a device memory pointer to a copy on the GPU.
    // Automatically updates the copy beforehand if the size does not match.
    // This memory will be freed on destruction, even if persistent is set.
    void* getDeviceCopy();
    // Updates the GPU copy of the image data.
    // The pointer may change, so make sure to use the new one.
    // This memory will be freed on destruction, even if persistent is set.
    void updateDeviceCopy();
    // Reads the image data from the device and updates the current image.
    // Throws an error if no GPU copy is present.
    void updateFromDevice();

    // Transform from ROI coordinates to image coordinates
    uint32_t fromROIxToImX(uint32_t x);
    // Transform from ROI coordinates to image coordinates
    uint32_t fromROIyToImY(uint32_t y);

    
private:
    // A copy of the image data that resides on the GPU.
    int m_gpuCopySize = 0;
    void* mpd_dataGPU = nullptr;

    ImageHandlerBase(); // No publically available default ctor

    // Returns the index of the keyword with the corresponding name.
    // If no corresponding keyword has been found, this method returns -1.
    int getKWindex(std::string name);
};




// Template declarations

template <typename U>
inline void ImageHandlerBase::setKeyword(int index, std::string name, U data)
{
    throw std::runtime_error("ImageHandlerBase::setKeyword: Only int64_t, double and string supported.");
}

template <>
inline void ImageHandlerBase::setKeyword(int index, std::string name, int64_t data)
{
    if (index >= mpImage->md->NBkw)
        throw std::runtime_error("ImageHandlerBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'L';
    kw.value.numl = data;
    mpImage->kw[index] = kw;
}

template <>
inline void ImageHandlerBase::setKeyword(int index, std::string name, double data)
{
    if (index >= mpImage->md->NBkw)
        throw std::runtime_error("ImageHandlerBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'D';
    kw.value.numf = data;
    mpImage->kw[index] = kw;
}

template <>
inline void ImageHandlerBase::setKeyword(int index, std::string name, std::string data)
{
    if (index >= mpImage->md->NBkw)
        throw std::runtime_error("ImageHandlerBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'S';
    int dataLen = std::min((int) data.length(), 16); // Max string length is 16
    std::strncpy(kw.value.valstr, data.c_str(), dataLen);
    sprintf(kw.comment, "%d", (int)data.length()); // Store the string length in the comment for easier access
    mpImage->kw[index] = kw;
}


template <typename U>
inline bool ImageHandlerBase::getKeyword(std::string name, U* dst)
{
    throw std::runtime_error("ImageHandlerBase::getKeyword: Only int64_t, double and string supported.");
}

template <>
inline bool ImageHandlerBase::getKeyword(std::string name, int64_t* dst)
{
    int kwIdx = getKWindex(name);
    if (kwIdx >= 0)
    {   
        IMAGE_KEYWORD kw = mpImage->kw[kwIdx];
        if (kw.type == 'L')
        {
            *dst = kw.value.numl;
            return true;
        }
    }
    return false;
}

template <>
inline bool ImageHandlerBase::getKeyword(std::string name, double* dst)
{
    int kwIdx = getKWindex(name);
    if (kwIdx >= 0)
    {   
        IMAGE_KEYWORD kw = mpImage->kw[kwIdx];
        if (kw.type == 'D')
        {
            *dst = kw.value.numf;
            return true;
        }
    }
    return false;
}

template <>
inline bool ImageHandlerBase::getKeyword(std::string name, std::string* dst)
{
    int kwIdx = getKWindex(name);
    if (kwIdx >= 0)
    {   
        IMAGE_KEYWORD kw = mpImage->kw[kwIdx];
        if (kw.type == 'S')
        {
            std::string tmp(kw.value.valstr);
            int strLen = std::stoi(kw.comment);
            *dst = tmp;
            while(dst->length() > strLen) // Only return the amount of chars stored in the comment
                dst->pop_back();
            return true;
        }
    }
    return false;
}

#endif  // IMAGEHANDLERBASE_HPP