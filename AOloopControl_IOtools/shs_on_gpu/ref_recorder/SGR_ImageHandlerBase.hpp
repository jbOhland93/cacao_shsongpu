// This class provides basic functionality for the more specific SGR_ImageHandler

#ifndef SGR_IMAGEHANDLERBASE_HPP
#define SGR_IMAGEHANDLERBASE_HPP

#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "ImageStreamIO/ImageStreamIO.h"
#include "../util/Rectangle.hpp"

class SGR_ImageHandlerBase
{
public:
    // The width of the image
    const uint32_t mWidth;
    // The height of the image
    const uint32_t mHeight;
    // The total number of pixels in the image
    const uint32_t mNumPx;

    ~SGR_ImageHandlerBase();

    // Returns the image object
    IMAGE* getImage() { return mpImage; }

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
    // The index of the device: -1 = CPU, 0 or greater = GPU index
    int32_t mDevice = -1;
    // If false, the image will be destoyed with the desturction of this instance
    bool mPersistent = false;
    // The region of interest. May be used by processing routines.
    Rectangle<uint32_t> mROI = Rectangle<uint32_t>(0, 0, 0, 0);

    // Ctor
    SGR_ImageHandlerBase(uint32_t width, uint32_t height, int32_t device = -1);

    // Transform from ROI coordinates to image coordinates
    uint32_t fromROIxToImX(uint32_t x);
    // Transform from ROI coordinates to image coordinates
    uint32_t fromROIyToImY(uint32_t y);

    
private:
    SGR_ImageHandlerBase(); // No publically available default ctor

    // Returns the index of the keyword with the corresponding name.
    // If no corresponding keyword has been found, this method returns -1.
    int getKWindex(std::string name);
};




// Template declarations

template <typename U>
inline void SGR_ImageHandlerBase::setKeyword(int index, std::string name, U data)
{
    throw std::runtime_error("SGR_ImageHandlerBase::setKeyword: Only int64_t, double and string supported.");
}

template <>
inline void SGR_ImageHandlerBase::setKeyword(int index, std::string name, int64_t data)
{
    if (index >= mpImage->md->NBkw)
        throw std::runtime_error("SGR_ImageHandlerBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'L';
    kw.value.numl = data;
    mpImage->kw[index] = kw;
}

template <>
inline void SGR_ImageHandlerBase::setKeyword(int index, std::string name, double data)
{
    if (index >= mpImage->md->NBkw)
        throw std::runtime_error("SGR_ImageHandlerBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'D';
    kw.value.numf = data;
    mpImage->kw[index] = kw;
}

template <>
inline void SGR_ImageHandlerBase::setKeyword(int index, std::string name, std::string data)
{
    if (index >= mpImage->md->NBkw)
        throw std::runtime_error("SGR_ImageHandlerBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'S';
    int dataLen = std::min((int) data.length(), 16); // Max string length is 16
    std::strncpy(kw.value.valstr, data.c_str(), dataLen);
    sprintf(kw.comment, "%d", (int)data.length()); // Store the string length in the comment for easier access
    mpImage->kw[index] = kw;
}


template <typename U>
inline bool SGR_ImageHandlerBase::getKeyword(std::string name, U* dst)
{
    throw std::runtime_error("SGR_ImageHandlerBase::getKeyword: Only int64_t, double and string supported.");
}

template <>
inline bool SGR_ImageHandlerBase::getKeyword(std::string name, int64_t* dst)
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
inline bool SGR_ImageHandlerBase::getKeyword(std::string name, double* dst)
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
inline bool SGR_ImageHandlerBase::getKeyword(std::string name, std::string* dst)
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

#endif  // SGR_IMAGEHANDLERBASE_HPP