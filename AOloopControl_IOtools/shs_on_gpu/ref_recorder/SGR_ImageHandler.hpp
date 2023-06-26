// This is a generic image handler, i.e. it should work for all datatypes.
// Generic specifications cannot be written into the compile unit (.cpp)
// Therefore, this is a long and inline heavy file. Sorry for that.

#ifndef SGR_IMAGEHANDLER_HPP
#define SGR_IMAGEHANDLER_HPP

#include "ImageStreamIO/ImageStreamIO.h"
#include <string>
#include <memory>
#include <vector>
#include <exception>
#include <limits>
#include <cstring>
#include <algorithm>
#include "../util/Rectangle.hpp"

#define spImageHandler(type) std::shared_ptr<SGR_ImageHandler<type>>
#define newImHandlerFrmIm(type, name, image) SGR_ImageHandler<type>::newHandlerfrmImage(name, image)

template <typename T>
class SGR_ImageHandler
{
public:
    const uint32_t mWidth;
    const uint32_t mHeight;
    const uint32_t mNumPx;

// ========== FACTORY FUNCTIONS ==========
    static spImageHandler(T) newImageHandler(
        std::string name,
        size_t width,
        size_t height,
        uint8_t numKeywords = 0,
        uint32_t circBufSize = 0);
    // Creates a new image of the same size, but converts the data.
    static spImageHandler(T) newHandlerfrmImage(
        std::string name,
        IMAGE* im,
        uint8_t numKeywords = 0,
        uint32_t circBufSize = 0);
    
    ~SGR_ImageHandler();

// ========== IMAGE HANDLING ==========    
    // Copies the data from im and converts to own datatype
    void cpy(IMAGE* im);
    // Calculates A-B and converts the result to own datatype
    void cpy_subtract(IMAGE* A, IMAGE* B);
    // Makes a binary image from the data in im
    // by comparing the pix values with thresh
    void cpy_thresh(IMAGE* im, double thresh);
    // Convolves A with kernel K stores result.
    // A has to have the same size as this image.
    // Values outside of A are considered 0.
    // Note: This is direct convolution, not made for speed.
    void cpy_convolve(IMAGE* A, IMAGE* K);

// ========== READING/WRITING MEMBER DATA ==========
    // Returns the image object
    IMAGE* getImage() { return &mImage; }
    // Reads the element at x/y from the last written buffer
    T read(uint32_t x, uint32_t y)
        { return mpBuffer[toROIy(y)*mWidth + toROIx(x)]; }
    // Writes the given element at teh x/y position into the write buffer
    void write(T e, uint32_t x, uint32_t y)
        { mpBuffer[toROIy(y)*mWidth + toROIx(x)] = e; }
    // Reads all the samples at x/y from the circular buffer
    std::vector<T> readCircularBufAt(uint32_t x, uint32_t y)
    {
        std::vector<T> v;
        T* cbBuf = (T*) mImage.CBimdata;
        for (int i = 0; i < mImage.md->CBsize; i++)
            v.push_back(cbBuf[i*mNumPx + y*mWidth + x]);
        return v;
    }
    // Setting a ROI
    void setROI(Rectangle<uint32_t> roi)
    {
        if (roi.x()+roi.w() >= mWidth || roi.y()+roi.h() >= mHeight)
            throw std::runtime_error("SGR_ImageHandler::setROI: out of range.");
        else
            mROI = roi;
    }
    // Setting a ROI
    void setROI(uint32_t x, uint32_t y, uint32_t w, uint32_t h)
        { setROI(Rectangle<uint32_t>(x,y,w,h)); }
    // Resetting the ROI
    void unsetROI() { mROI = Rectangle<uint32_t>(0,0, mWidth, mHeight); }
    Rectangle<uint32_t> getROI() { return mROI; }
    // Makes the image stream stay after destruction
    void setPersistent(bool persistent) { mPersistent = persistent; }
    // Setting a keyword
    template <typename U>
    void setKeyword(int index, std::string name, U data)
    { throw std::runtime_error("SGR_ImageHandler::setKeyword: Only int64_t, double and string supported."); }
    //template<>
    void setKeyword(int index, std::string name, int64_t data)
    {
        if (index >= mImage.md->NBkw)
            throw std::runtime_error("SGR_ImageHandler::setKeyword: Index is larger than the number of available keywords.");
        IMAGE_KEYWORD kw;
        std::strncpy(kw.name, name.c_str(), name.length());
        kw.type = 'L';
        kw.value.numl = data;
        mImage.kw[index] = kw;
    }
    //template<>
    void setKeyword(int index, std::string name, double data)
    {
        if (index >= mImage.md->NBkw)
            throw std::runtime_error("SGR_ImageHandler::setKeyword: Index is larger than the number of available keywords.");
        IMAGE_KEYWORD kw;
        std::strncpy(kw.name, name.c_str(), name.length());
        kw.type = 'D';
        kw.value.numf = data;
        mImage.kw[index] = kw;
    }
    //template<>
    void setKeyword(int index, std::string name, std::string data)
    {
        if (index >= mImage.md->NBkw)
            throw std::runtime_error("SGR_ImageHandler::setKeyword: Index is larger than the number of available keywords.");
        IMAGE_KEYWORD kw;
        std::strncpy(kw.name, name.c_str(), name.length());
        kw.type = 'S';
        std::strncpy(kw.value.valstr, data.c_str(), std::min(16, (int)data.length()));
        mImage.kw[index] = kw;
    }
    // Reading a keyword
    template <typename U>
    bool getKeyword(std::string name, U* dst)
    { throw std::runtime_error("SGR_ImageHandler::getKeyword: Only int64_t, double and string supported."); }
    bool getKeyword(std::string name, int64_t* dst)
    {
        int kwIdx = getKWindex(name);
        if (kwIdx >= 0)
        {   
            IMAGE_KEYWORD kw = mImage.kw[kwIdx];
            if (kw.type == 'L')
            {
                *dst = kw.value.numl;
                return true;
            }
        }
        return false;
    }
    //template<>
    bool getKeyword(std::string name, double* dst)
    {
        int kwIdx = getKWindex(name);
        if (kwIdx >= 0)
        {   
            IMAGE_KEYWORD kw = mImage.kw[kwIdx];
            if (kw.type == 'D')
            {
                *dst = kw.value.numf;
                return true;
            }
        }
        return false;
    }
    //template<>
    bool getKeyword(std::string name, std::string* dst)
    {
        int kwIdx = getKWindex(name);
        if (kwIdx >= 0)
        {   
            IMAGE_KEYWORD kw = mImage.kw[kwIdx];
            if (kw.type == 'S')
            {
                std::string tmp(kw.value.valstr);
                *dst = tmp;
                return true;
            }
        }
        return false;
    }

// ========== OPERATIONS ==========
    // Updates the image and gets the new write buffer
    void updateWrittenImage()
        { ImageStreamIO_UpdateIm(&mImage); }
    // Get the number of pixels in the current ROI
    uint32_t getPxInROI() { return mROI.w() * mROI.h(); }
    // Sums all the pixels in the current ROI
    double getSumOverROI()
    {
        double sum = 0;
        for (uint32_t ix = 0; ix < mROI.w(); ix++)
            for (uint32_t iy = 0; iy < mROI.h(); iy++)
                sum += (double) read(ix, iy);
        return sum;
    }
    // Gets the maximum pixel value in the current ROI
    T getMaxInROI(uint32_t* maxposX = nullptr, uint32_t* maxposY= nullptr)
    {
        T max = std::numeric_limits<T>::min();
        T cur;
        for (uint32_t ix = 0; ix < mROI.w(); ix++)
            for (uint32_t iy = 0; iy < mROI.h(); iy++)
            {
                cur = read(ix, iy);
                if (cur > max)
                {
                    max = cur;
                    if (maxposX != nullptr) *maxposX = ix;
                    if (maxposY != nullptr) *maxposY = iy;
                }
            }
        return max;
    }
    // Erodes the image inside the ROI
    // Returns the number of remaining valid pixels
    // Appends the coordinates of the dissolved particles to d
    uint32_t erode(std::vector<Point<uint32_t>>* d = nullptr);
    
private:
    IMAGE mImage;
    T* mpBuffer = nullptr;
    bool mPersistent = false;
    Rectangle<uint32_t> mROI = Rectangle<uint32_t>(0, 0, 0, 0);
// ========== CONSTRUCTORS ==========
    SGR_ImageHandler(); // No publically available default ctor
    SGR_ImageHandler(
        std::string name,
        uint32_t width,
        uint32_t height,
        uint8_t atype,
        uint8_t numKeywords,
        uint32_t circBufSize = 10);

// ========== HELPER FUNCTIONS ==========
    // Transform from ROI coordinates to image coordinates
    uint32_t toROIx(uint32_t x)
    {
        if (x >= mROI.w()) // x is uint3_t, thus always > 0
            throw std::runtime_error("SGR_ImageHandler::toROIx: x is out of range.");
        else
            return x + mROI.x();
    }
    // Transform from image coordinates to ROI coordinates
    uint32_t toROIy(uint32_t y)
    {
        if (y >= mROI.h()) // y is uint3_t, thus always > 0
            throw std::runtime_error("SGR_ImageHandler::toROIx: x is out of range.");
        else
            return y + mROI.y();
    }
    // Conversion for foreign element access
    template <typename U>
    U cvtElmt(void* ptr, uint8_t atype, uint32_t index)
    {
        switch(atype) {
            case _DATATYPE_UINT8: return (U)(*(((uint8_t*)ptr)+index));
            case _DATATYPE_INT8: return (U)(*(((int8_t*)ptr)+index));
            case _DATATYPE_UINT16: return (U)(*(((uint16_t*)ptr)+index));
            case _DATATYPE_INT16: return (U)(*(((int16_t*)ptr)+index));
            case _DATATYPE_UINT32: return (U)(*(((uint32_t*)ptr)+index));
            case _DATATYPE_INT32: return (U)(*(((int32_t*)ptr)+index));
            case _DATATYPE_UINT64: return (U)(*(((uint64_t*)ptr)+index));
            case _DATATYPE_INT64: return (U)(*(((int64_t*)ptr)+index));
            case _DATATYPE_FLOAT: return (U)(*(((float*)ptr)+index));
            case _DATATYPE_DOUBLE: return (U)(*(((double*)ptr)+index));
            default:
            throw std::runtime_error("SGR_ImageHandler::cvtElmt: No case for this data type.\n");
        }
    }
    T cvtElmt(void* ptr, uint8_t atype, uint32_t index)
    {
        return cvtElmt<T>(ptr, atype, index);
    }
    int getKWindex(std::string name);
};

// ######################################
// Implementation
// ######################################

// ===== specifications of factory functions =====

// Default implementation, throws an error
template <typename T>
inline spImageHandler(T) SGR_ImageHandler<T>::newImageHandler(
    std::string name,
    size_t width,
    size_t height,
    uint8_t numKeywords,
    uint32_t circBufSize)
{
    throw std::runtime_error(
        "SGR_ImageHandler<T>::newImageHandler: Type not supported.");
    return nullptr;
}

// Defining specific factory functions via a makro ... way shorter!
#define IH_FACTORY(type, atype)                                         \
template <>                                                             \
inline spImageHandler(type) SGR_ImageHandler<type>::newImageHandler(    \
    std::string name,                                                   \
    size_t width,                                                       \
    size_t height,                                                      \
    uint8_t numKeywords,                                                \
    uint32_t circBufSize)                                               \
{                                                                       \
    spImageHandler(type) sp(new SGR_ImageHandler<type>(                 \
        name,                                                           \
        width,                                                          \
        height,                                                         \
        atype,                                                          \
        numKeywords,                                                    \
        circBufSize));                                                  \
    return sp;                                                          \
}
IH_FACTORY(uint8_t, _DATATYPE_UINT8);
IH_FACTORY(int8_t, _DATATYPE_INT8);
IH_FACTORY(uint16_t, _DATATYPE_UINT16);
IH_FACTORY(int16_t, _DATATYPE_INT16);
IH_FACTORY(uint32_t, _DATATYPE_UINT32);
IH_FACTORY(int32_t, _DATATYPE_INT32);
IH_FACTORY(uint64_t, _DATATYPE_UINT64);
IH_FACTORY(int64_t, _DATATYPE_INT64);
IH_FACTORY(float, _DATATYPE_FLOAT);
IH_FACTORY(double, _DATATYPE_DOUBLE);
// ==END== specifications of factory functions ==END==

template <typename T>
inline spImageHandler(T) SGR_ImageHandler<T>::newHandlerfrmImage(
        std::string name,
        IMAGE* im,
        uint8_t numKeywords,
        uint32_t circBufSize)
{
    uint32_t* size = im->md->size;
    spImageHandler(T) spIH = 
        newImageHandler(name, size[0], size[1], numKeywords, circBufSize);
    spIH->cpy(im);
    return spIH;
}

// ===== Destructor =====
template <typename T>
inline SGR_ImageHandler<T>::~SGR_ImageHandler()
{
    if (!mPersistent)
        ImageStreamIO_destroyIm(&mImage);
}

// ===== specifications for image handling =====

template <typename T>
inline void SGR_ImageHandler<T>::cpy(IMAGE* im)
{
    // Check image sizes
    uint32_t* size = im->md->size;
    if (size[0] != mWidth || size[1] != mHeight)
        throw std::runtime_error("SGR_ImageHandler::cpy: Incompatible size of input image.\n");

    // Prepare read-buffer
    void* readBuffer;
    ImageStreamIO_readLastWroteBuffer(im, &readBuffer);

    // Convert
    uint8_t atype = im->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mpBuffer[i] = cvtElmt(readBuffer, atype, i);
    
    updateWrittenImage();
}

template <typename T>
inline void SGR_ImageHandler<T>::cpy_subtract(IMAGE* A, IMAGE* B)
{
    // Check image sizes
    uint32_t* sizeA = A->md->size;
    uint32_t* sizeB = B->md->size;
    if (sizeA[0] != mWidth
        || sizeB[0] != mWidth
        || sizeA[1] != mHeight
        || sizeB[1] != mHeight)
        throw std::runtime_error("SGR_ImageHandler::cpy_subtract: Incompatible size of input image(s).\n");

    // Prepare buffers
    void* readBufferA;
    ImageStreamIO_readLastWroteBuffer(A, &readBufferA);
    void* readBufferB;
    ImageStreamIO_readLastWroteBuffer(B, &readBufferB);

    // Convert and subtract
    uint8_t atypeA = A->md->datatype;
    uint8_t atypeB = B->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mpBuffer[i] = cvtElmt(readBufferA, atypeA, i)
            - cvtElmt(readBufferB, atypeB, i);

    updateWrittenImage();
}

template <typename T>
inline void SGR_ImageHandler<T>::cpy_thresh(IMAGE* im, double thresh)
{
    // Check image sizes
    uint32_t* size = im->md->size;
    if (size[0] != mWidth || size[1] != mHeight)
        throw std::runtime_error("SGR_ImageHandler::cpy_thresh: Incompatible size of input image.\n");

    // Prepare buffer
    void* readBuffer;
    ImageStreamIO_readLastWroteBuffer(im, &readBuffer);

    // Convert
    uint8_t atype = im->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mpBuffer[i] = cvtElmt<double>(readBuffer, atype, i) > thresh;
    
    updateWrittenImage();
}

template <typename T>
inline void SGR_ImageHandler<T>::cpy_convolve(IMAGE* A, IMAGE* K)
{
    // Check image size
    uint32_t* size = A->md->size;
    if (size[0] != mWidth || size[1] != mHeight)
        throw std::runtime_error("SGR_ImageHandler::cpy_convolve: Incompatible size of input A.\n");

    // Prepare buffer
    void* bA;
    ImageStreamIO_readLastWroteBuffer(A, &bA);
    void* bK;
    ImageStreamIO_readLastWroteBuffer(K, &bK);

    // Collect metadata
    uint8_t tA = A->md->datatype;   // Kernel datatype for casting
    uint8_t tK = K->md->datatype;   // Kernel datatype for casting
    int32_t kw = K->md->size[0];   // Kernel width
    int32_t kh = K->md->size[1];   // Kernel height
    int32_t kcX = floor(kw/2);     // Kernel center X
    int32_t kcY = floor(kh/2);     // Kernel center Y

    // Prepare fields
    float r;    // Convolution result
    float k;    // Value of the kernel
    int32_t bi; // The buffer index, calculated from x and y
    
    // Do convolution
    for (int32_t ix = 0; ix < mWidth; ix++)
        for (int32_t iy = 0; iy < mHeight; iy++)
        {
            r = 0;
            for (int32_t kx = -kcX; kx < kw-kcX; kx++)
                for (int32_t ky = -kcY; ky < kh-kcY; ky++)
                {
                    if (ix+kx >= 0 && ix+kx < mWidth         // Only inside of A
                        && iy+ky >= 0 && iy+ky < mHeight)
                    {
                        bi = (ky+kcY)*kw + kx+kcX;           // Kernel index
                        k = cvtElmt<float>(bK, tK, bi);      // Kernel value
                        bi = (iy+ky)*mWidth + ix+kx;         // Image index
                        r += cvtElmt<float>(bA, tA, bi) * k; // Convolution value
                    }
                }
            mpBuffer[iy*mWidth + ix] = (T) r;
        }
    
    updateWrittenImage();
}


// Implementation of ctor
template<typename T>
inline SGR_ImageHandler<T>::SGR_ImageHandler(
        std::string name,
        uint32_t width,
        uint32_t height,
        uint8_t atype,
        uint8_t numKeywords,
        uint32_t circBufSize)
        :
        mWidth(width),
        mHeight(height),
        mNumPx(width*height),
        mROI(0,0,width,height)
{
    int naxis = 2;
    uint32_t * imsize = new uint32_t[naxis]();
    imsize[0] = width;
    imsize[1] = height;
    int shared = 1; // image will be in shared memory
    ImageStreamIO_createIm(&mImage,
                            name.c_str(),
                            naxis,
                            imsize,
                            atype,
                            shared,
                            numKeywords,
                            circBufSize);
    delete imsize;
    ImageStreamIO_writeBuffer(&mImage, (void**) &mpBuffer);
}

template <typename T>
uint32_t SGR_ImageHandler<T>::erode(std::vector<Point<uint32_t>>* d)
{
    uint32_t remainingPixels = 0;
    int x;
    int y;
    int neighbourIndex;
    int neighbours;
    // Precalculate the x/y offsets of the surrounding pixels
    int nOffX[8];
    int nOffY[8];
    for (int j = 0; j < 8; j++)
    {
        nOffX[j] = round(cos(j*M_PI/4.));
        nOffY[j] = round(sin(j*M_PI/4.));
    }

    // Copy the read frame to the write buffer
    for (int i = 0; i < mNumPx; i++)
        mpBuffer[i] = mpBuffer[i];

    // Erode the ROI
    for (uint32_t ix = 0; ix < mROI.w(); ix++)
        for (uint32_t iy = 0; iy < mROI.h(); iy++)
        {   
            x = toROIx(ix);
            y = toROIy(iy);
            // Only consider pixel if it is greater than 0
            if (mpBuffer[y*mWidth + x] > 0)
            {   // Count the neighbours
                neighbours = 0;
                for (int j = 0; j < 8; j++)
                {
                    if (x+nOffX[j] >= 0 && x+nOffX[j] < mWidth &&
                        y+nOffY[j] >= 0 && y+nOffY[j] < mHeight)
                    {
                        neighbourIndex = (y+nOffY[j])*mWidth + x + nOffX[j];
                        if (mpBuffer[neighbourIndex] > 0)
                            neighbours++;
                    }
                }
                if (neighbours == 0)
                {   // Standalonepixel - Dissolving particle.
                    if (d != nullptr)
                        d->push_back(
                            Point<uint32_t>(x,y,mpBuffer[y*mWidth + x], true));
                    // Set to zero.
                    mpBuffer[y*mWidth + x] = 0;
                }
                else if (neighbours < 5)
                    // Edgepixel - set to zero.
                    mpBuffer[y*mWidth + x] = 0;
                else
                    // Embedded pixel, leave as it is.
                    remainingPixels++;
            }
        }
    
    updateWrittenImage();
    return remainingPixels;
}

template <typename T>
int SGR_ImageHandler<T>::getKWindex(std::string name)
{
    for (int i = 0; i < mImage.md->NBkw; i++)
        if (name == mImage.kw[i].name)
            return i;
    return 0;
}

#endif // SGR_IMAGEHANDLER_HPP