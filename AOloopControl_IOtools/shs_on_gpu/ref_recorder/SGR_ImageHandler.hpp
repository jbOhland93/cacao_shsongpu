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
        size_t height);
    // Creates a new image of the same size, but converts the data.
    static spImageHandler(T) newHandlerfrmImage(
        std::string name,
        IMAGE* im);

// ========== IMAGE HANDLING ==========    
    // Copies the data from im and converts to own datatype
    void cpy(IMAGE* im);
    // Calculates A-B and converts the result to own datatype
    void cpy_subtract(IMAGE* A, IMAGE* B);
    // Makes a binary image from the data in im
    // by comparing the pix values with thresh
    void cpy_thresh(IMAGE* im, double thresh);

// ========== READING/WRITING MEMBER DATA ==========
    // Updates the read buffer to the last written frame
    IMAGE* getImage() {return &mImage;}
    
    // Reads the element at x/y from the last written buffer
    T read(uint32_t x, uint32_t y)
    {
        if (mpReadBuffer == nullptr)
        {
            printf("IS NULL!\n");
            updateReadBuffer();
        }
        return mpReadBuffer[toROIy(y)*mWidth + toROIx(x)];
    }
    // Writes the given element at teh x/y position into the write buffer
    void write(T e, uint32_t x, uint32_t y)
        { mpWriteBuffer[toROIy(y)*mWidth + toROIx(x)] = e; }
    // Setting a ROI
    void setROI(Rectangle<uint32_t> roi)
    {
        if (roi.x()+roi.w() >= mWidth ||
            roi.y()+roi.h() >= mHeight)
            throw std::runtime_error("SGR_ImageHandler::setROI: out of range.");
        else
            mROI = roi;
    }
    // Setting a ROI
    void setROI(uint32_t x, uint32_t y, uint32_t w, uint32_t h)
        { setROI(Rectangle<uint32_t>(x,y,w,h)); }
    // Resetting the ROI
    void unsetROI() { mROI = Rectangle<uint32_t>(0,0, mWidth, mHeight); }

// ========== OPERATIONS ==========
    // Updates the reading buffer to the lates written buffer
    void updateReadBuffer()
        { ImageStreamIO_readLastWroteBuffer(&mImage, (void**)&mpReadBuffer); }
    // Updates the image and gets the new write buffer
    void updateWrittenImage()
    {
        ImageStreamIO_UpdateIm(&mImage);
        ImageStreamIO_writeBuffer(&mImage, (void**) &mpWriteBuffer);
        updateReadBuffer();
    }
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
    T getMaxInROI()
    {
        T max = std::numeric_limits<T>::min();
        T cur;
        for (uint32_t ix = 0; ix < mROI.w(); ix++)
            for (uint32_t iy = 0; iy < mROI.h(); iy++)
            {
                cur = read(ix, iy);
                if (cur > max)
                    max = cur;
            }
        return max;
    }
    // Erodes the image inside the ROI
    // Returns the number of remaining valid pixels
    // Appends the coordinates of the dissolved particles to d
    uint32_t erode(std::vector<Point<uint32_t>>* d = nullptr);
    
private:
    IMAGE mImage;
    T* mpReadBuffer = nullptr;
    T* mpWriteBuffer = nullptr;
    Rectangle<uint32_t> mROI = Rectangle<uint32_t>(0, 0, 0, 0);
// ========== CONSTRUCTORS ==========
    SGR_ImageHandler(); // No publically available default ctor
    SGR_ImageHandler(
        std::string name,
        uint32_t width,
        uint32_t height,
        uint8_t atype);

// ========== HELPER FUNCTIONS ==========
    // Transform from ROI coordinates to image coordinates
    uint32_t toROIx(uint32_t x)
    {
        if (x < 0 || x >= mROI.w())
            throw std::runtime_error("SGR_ImageHandler::toROIx: x is out of range.");
        else
            return x + mROI.x();
    }
    // Transform from image coordinates to ROI coordinates
    uint32_t toROIy(uint32_t y)
    {
        if (y < 0 || y >= mROI.h())
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
    size_t height)
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
    size_t height)                                                      \
{                                                                       \
    spImageHandler(type) sp(new SGR_ImageHandler<type>(                 \
        name,                                                           \
        width,                                                          \
        height,                                                         \
        atype));                                                        \
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
        IMAGE* im)
{
    uint32_t* size = im->md->size;
    spImageHandler(T) spIH = newImageHandler(name, size[0], size[1]);
    spIH->cpy(im);
    return spIH;
}

template <typename T>
inline void SGR_ImageHandler<T>::cpy(IMAGE* im)
{
    // Check image sizes
    uint32_t* size = im->md->size;
    if (size[0] != mWidth || size[1] != mHeight)
        throw std::runtime_error("SGR_ImageHandler::cpy: Incompatible size of input image.\n");

    // Prepare read-buffer
    ImageStreamIO_writeBuffer(&mImage, (void**) &mpWriteBuffer);
    void* readBuffer;
    ImageStreamIO_readLastWroteBuffer(im, &readBuffer);

    // Convert
    uint8_t atype = im->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mpWriteBuffer[i] = cvtElmt(readBuffer, atype, i);
    
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
    ImageStreamIO_writeBuffer(&mImage, (void**) &mpWriteBuffer);
    void* readBufferA;
    ImageStreamIO_readLastWroteBuffer(A, &readBufferA);
    void* readBufferB;
    ImageStreamIO_readLastWroteBuffer(B, &readBufferB);

    // Convert and subtract
    uint8_t atypeA = A->md->datatype;
    uint8_t atypeB = B->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mpWriteBuffer[i] = cvtElmt(readBufferA, atypeA, i) - cvtElmt(readBufferB, atypeB, i);

    updateWrittenImage();
}

template <typename T>
inline void SGR_ImageHandler<T>::cpy_thresh(IMAGE* im, double thresh)
{
    // Check image sizes
    uint32_t* size = im->md->size;
    if (size[0] != mWidth || size[1] != mHeight)
        throw std::runtime_error("SGR_ImageHandler::cpy: Incompatible size of input image.\n");

    // Prepare read-buffer
    ImageStreamIO_writeBuffer(&mImage, (void**) &mpWriteBuffer);
    void* readBuffer;
    ImageStreamIO_readLastWroteBuffer(im, &readBuffer);

    // Convert
    uint8_t atype = im->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mpWriteBuffer[i] = cvtElmt<double>(readBuffer, atype, i) > thresh;
    
    updateWrittenImage();
}

// Implementation of ctor
template<typename T>
inline SGR_ImageHandler<T>::SGR_ImageHandler(
        std::string name,
        uint32_t width,
        uint32_t height,
        uint8_t atype)
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
    int NBkw = 0; // No keywords for now. Do this later.
    int circBufSize = 10;
    ImageStreamIO_createIm(&mImage,
                            name.c_str(),
                            naxis,
                            imsize,
                            atype,
                            shared,
                            NBkw,
                            circBufSize);
    delete imsize;
    ImageStreamIO_writeBuffer(&mImage, (void**) &mpWriteBuffer);
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
        mpWriteBuffer[i] = mpReadBuffer[i];

    // Erode the ROI
    for (uint32_t ix = 0; ix < mROI.w(); ix++)
        for (uint32_t iy = 0; iy < mROI.h(); iy++)
        {   
            x = toROIx(ix);
            y = toROIy(iy);
            // Only consider pixel if it is greater than 0
            if (mpWriteBuffer[y*mWidth + x] > 0)
            {   // Count the neighbours
                neighbours = 0;
                for (int j = 0; j < 8; j++)
                {
                    if (x+nOffX[j] >= 0 && x+nOffX[j] < mWidth &&
                        y+nOffY[j] >= 0 && y+nOffY[j] < mHeight)
                    {
                        neighbourIndex = (y+nOffY[j])*mWidth + x + nOffX[j];
                        if (mpWriteBuffer[neighbourIndex] > 0)
                            neighbours++;
                    }
                }
                if (neighbours == 0)
                {   // Standalonepixel - Dissolving particle.
                    if (d != nullptr)
                        d->push_back(
                            Point<uint32_t>(x,y,mpReadBuffer[y*mWidth + x], true));
                    // Set to 0
                    mpWriteBuffer[y*mWidth + x] = 0;
                }
                else if (neighbours < 4)
                    // Edgepixel - set to zero.
                    mpWriteBuffer[y*mWidth + x] = 0;
                else
                    // Embedded pixel, leave as it is.
                    remainingPixels++;
            }
        }
    
    updateWrittenImage();
    return remainingPixels;
}

#endif // SGR_IMAGEHANDLER_HPP