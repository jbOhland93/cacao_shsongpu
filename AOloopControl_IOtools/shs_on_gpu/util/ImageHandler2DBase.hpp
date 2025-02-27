// A class for handling 2D image streams with support of slices in the 3rd dimension.
// This class provides basic functionality for the more specific SGR_ImageHandler

#ifndef IMAGEHANDLERBASE_HPP
#define IMAGEHANDLERBASE_HPP

#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include "../util/Rectangle.hpp"

#ifdef __cplusplus
extern "C"
{
#endif
#include "../../../../../src/CommandLineInterface/CLIcore.h"
#ifdef __cplusplus
}
#endif

#define spIHBase std::shared_ptr<ImageHandler2DBase>
class ImageHandler2DBase
{
public:
    // The width of the image
    const uint32_t mWidth;
    // The height of the image
    const uint32_t mHeight;
    // The depth of the image
    const uint32_t mDepth;
    // The total number of pixels in the image
    const uint32_t mNumPx;

    ~ImageHandler2DBase();

    // Returns the image object
    IMAGE* getImage() { return mp_image; }
    // Returns the image array size in memory
    size_t getBufferSize() { return mp_image->md->imdatamemsize; }
    // Returns the current image counter
    uint64_t getCnt0() { return mp_image->md->cnt0; }

    // Makes the image stream stay after destruction
    void setPersistent(bool persistent) { mPersistent = persistent; }
    cudaError_t mapImForGPUaccess();

    // Setting the current slice
    void setSlice(uint32_t sliceIndex);
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
    void updateWrittenImage() { ImageStreamIO_UpdateIm(mp_image); }
    // Waits for the next frame
    bool waitForNextFrame(int timeout_us = 1e6);

    // Saves the image to the data directory of the given fps
    void saveToFPSdataDir(FUNCTION_PARAMETER_STRUCT* fps, std::string fname);
    
protected:
    // The image, managed by this class
    IMAGE* mp_image;
    void* mp_h_imData = nullptr;
    uint64_t m_dataSize = 0;
    uint64_t m_currentSlice = 0;
    // If false, the image will be destoyed with the desturction of this instance
    bool mPersistent = false;
    // The cnt0 value of the last frame that has been copied to the GPU.
    uint64_t mCnt0deviceCopy = std::numeric_limits<uint64_t>::max();
    // The region of interest. May be used by processing routines.
    Rectangle<uint32_t> mROI = Rectangle<uint32_t>(0, 0, 0, 0);
    // Index of the semaphore used by this image handler
    uint_fast16_t m_semaphoreIndex = 0;

    // Ctor
    ImageHandler2DBase(uint32_t width, uint32_t height, uint32_t depth = 1);
    // Updates locally stored image data for quick access
    // Has to be called by decendants
    void updateImMetadata();

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

    // Get the size vector of this class
    std::vector<uint32_t> getSizeVector();
    
    // Convert the size of an image to a vector of exactly 3 elements
    // Throws an error if the image features a higher dimensionality
    static std::vector<uint32_t> imSizeToVector(IMAGE* im)
    {
        uint8_t naxis = im->md->naxis;
        if (naxis > 3)
            throw std::runtime_error("ImageHandler2DBase::imSizeToVector: dimensionality cannot be greater than 3.");
        uint32_t* size = im->md->size;

        std::vector<uint32_t> sVec;
        sVec.push_back(size[0]);
        sVec.push_back(naxis > 1 ? size[1] : 1);
        sVec.push_back(naxis > 2 ? size[2] : 1);

        return sVec;
    }

    
private:
    // A copy of the image data that resides on the GPU.
    int m_gpuCopySize = 0;
    void* mp_d_imData = nullptr;

    ImageHandler2DBase(); // No publically available default ctor

    // Returns the index of the keyword with the corresponding name.
    // If no corresponding keyword has been found, this method returns -1.
    int getKWindex(std::string name);
};




// Template declarations

template <typename U>
inline void ImageHandler2DBase::setKeyword(int index, std::string name, U data)
{
    throw std::runtime_error("ImageHandler2DBase::setKeyword: Only int64_t, double and string supported.");
}

template <>
inline void ImageHandler2DBase::setKeyword(int index, std::string name, int64_t data)
{
    if (index >= mp_image->md->NBkw)
        throw std::runtime_error("ImageHandler2DBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'L';
    kw.value.numl = data;
    mp_image->kw[index] = kw;
}

template <>
inline void ImageHandler2DBase::setKeyword(int index, std::string name, double data)
{
    if (index >= mp_image->md->NBkw)
        throw std::runtime_error("ImageHandler2DBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'D';
    kw.value.numf = data;
    mp_image->kw[index] = kw;
}

template <>
inline void ImageHandler2DBase::setKeyword(int index, std::string name, std::string data)
{
    if (index >= mp_image->md->NBkw)
        throw std::runtime_error("ImageHandler2DBase::setKeyword: Index is larger than the number of available keywords.");
    IMAGE_KEYWORD kw;
    std::strncpy(kw.name, name.c_str(), name.length());
    kw.type = 'S';
    int dataLen = std::min((int) data.length(), 16); // Max string length is 16
    std::strncpy(kw.value.valstr, data.c_str(), dataLen);
    sprintf(kw.comment, "%d", (int)data.length()); // Store the string length in the comment for easier access
    mp_image->kw[index] = kw;
}


template <typename U>
inline bool ImageHandler2DBase::getKeyword(std::string name, U* dst)
{
    throw std::runtime_error("ImageHandler2DBase::getKeyword: Only int64_t, double and string supported.");
}

template <>
inline bool ImageHandler2DBase::getKeyword(std::string name, int64_t* dst)
{
    int kwIdx = getKWindex(name);
    if (kwIdx >= 0)
    {   
        IMAGE_KEYWORD kw = mp_image->kw[kwIdx];
        if (kw.type == 'L')
        {
            *dst = kw.value.numl;
            return true;
        }
    }
    return false;
}

template <>
inline bool ImageHandler2DBase::getKeyword(std::string name, double* dst)
{
    int kwIdx = getKWindex(name);
    if (kwIdx >= 0)
    {   
        IMAGE_KEYWORD kw = mp_image->kw[kwIdx];
        if (kw.type == 'D')
        {
            *dst = kw.value.numf;
            return true;
        }
    }
    return false;
}

template <>
inline bool ImageHandler2DBase::getKeyword(std::string name, std::string* dst)
{
    int kwIdx = getKWindex(name);
    if (kwIdx >= 0)
    {   
        IMAGE_KEYWORD kw = mp_image->kw[kwIdx];
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