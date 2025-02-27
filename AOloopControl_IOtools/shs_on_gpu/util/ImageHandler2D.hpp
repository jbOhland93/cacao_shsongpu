// This is a generic image handler, i.e. it should work for all datatypes.
// All functionality is implemented with 2D image streams in mind
// with support of slices in the 3rd dimension.
// Generic specifications cannot be written into the compile unit (.cpp)
// Therefore, this is a long and inline heavy file. Sorry for that.

#ifndef IMAGEHANDLER_HPP
#define IMAGEHANDLER_HPP

#include "ImageHandler2DBase.hpp"
#include "atypeUtil.hpp"
#include <memory>
#include <vector>
#include <limits>
#include <stdexcept>

#define spImHandler2D(type) std::shared_ptr<ImageHandler2D<type>>
#define newImHandler2DFrmIm(type, name, image) ImageHandler2D<type>::newHandler2DfrmImage(name, image)

template <typename T>
class ImageHandler2D : public ImageHandler2DBase
{
public:

// ========== FACTORY FUNCTIONS ==========
    static spImHandler2D(T) newImageHandler2D(
        std::string name,
        size_t width,
        size_t height,
        size_t depth = 1,
        uint8_t numKeywords = 0,
        uint32_t circBufSize = 0);
    // Creates a new image of the same size, but converts the data.
    static spImHandler2D(T) newHandler2DfrmImage(
        std::string name,
        IMAGE* im,
        uint8_t numKeywords = 0,
        uint32_t circBufSize = 0);
    // Creates an image handler that operates on an already existing image.
    // Throws an error if the generic type does not match the image type.
    static spImHandler2D(T) newHandler2DAdoptImage(std::string imName);

// ========== IMAGE HANDLING ==========    
    // Copies the data from im and converts to own datatype
    void cpy(IMAGE* im, bool triggerUpdate = true);
    // Calculates A-B and converts the result to own datatype
    void cpy_subtract(IMAGE* A, IMAGE* B, bool triggerUpdate = true);
    // Makes a binary image from the data in im
    // by comparing the pix values with thresh
    void cpy_thresh(IMAGE* im, double thresh, bool triggerUpdate = true);
    // Convolves A with kernel K stores result.
    // A has to have the same size as this image.
    // Values outside of A are considered 0.
    // Note: This is direct convolution, not made for speed.
    void cpy_convolve(IMAGE* A, IMAGE* K, bool triggerUpdate = true);

// ========== READING/WRITING MEMBER DATA ==========
    // Returns the write buffer
    T* getWriteBuffer() { return mp_data; }
    // Returns the current data copy that resides in device emory
    // Updates the copy if the image size does not match prior to returning.
    // This memory will be freed on destruction, even if persistent is set.
    T* getGPUCopy() { return (T*) getDeviceCopy(); }
    // Updates the GPU copy and returns the new device pointer.
    // This memory will be freed on destruction, even if persistent is set.
    T* updateGPUCopy() { updateDeviceCopy(); return getGPUCopy(); }
    // Reads the image data from the GPU and updates the current image.
    // Throws an error if no GPU copy is present.
    void updateFromGPU() { updateFromDevice(); }
    // Reads the element at x/y from the last written buffer
    T read(uint32_t x, uint32_t y)
        { return mp_data[m_currentSlice*mWidth*mHeight + fromROIyToImY(y)*mWidth + fromROIxToImX(x)]; }
    // Writes the given element at teh x/y position into the write buffer
    void write(T e, uint32_t x, uint32_t y)
        { mp_data[m_currentSlice*mWidth*mHeight + fromROIyToImY(y)*mWidth + fromROIxToImX(x)] = e; }
    // Reads all the samples at x/y from the circular buffer
    std::vector<T> readCircularBufAt(uint32_t x, uint32_t y)
    {
        std::vector<T> v;
        uint32_t xIm = fromROIxToImX(x);
        uint32_t yIm = fromROIyToImY(y);
        T* cbBuf = (T*) mp_image->CBimdata;
        for (int i = 0; i < mp_image->md->CBsize; i++)
            v.push_back(cbBuf[i*mNumPx + m_currentSlice*mWidth*mHeight + yIm*mWidth + xIm]);
        return v;
    }
    

// ========== OPERATIONS ==========
    // Sums all the pixels in the current ROI of the current slice
    double getSumOverROI()
    {
        double sum = 0;
        for (uint32_t ix = 0; ix < mROI.w(); ix++)
            for (uint32_t iy = 0; iy < mROI.h(); iy++)
                sum += (double) read(ix, iy);
        return sum;
    }
    // Gets the maximum pixel value in the current ROI of the current slice
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
    // Gets the minimum pixel value in the current ROI of the current slice
    T getMinInROI(uint32_t* minposX = nullptr, uint32_t* minposY= nullptr)
    {
        T min = std::numeric_limits<T>::max();
        T cur;
        for (uint32_t ix = 0; ix < mROI.w(); ix++)
            for (uint32_t iy = 0; iy < mROI.h(); iy++)
            {
                cur = read(ix, iy);
                if (cur < min)
                {
                    min = cur;
                    if (minposX != nullptr) *minposX = ix;
                    if (minposY != nullptr) *minposY = iy;
                }
            }
        return min;
    }
    // Erodes the image inside the ROI at the current slice
    // Returns the number of remaining valid pixels in the current slice
    // neighboursToSurvive: selfexplanatory
    // inPlace: if true, one eroded pixel will affect the survival of the next one
    // d: the coordinates to dissolved particles are appended to this vector
    uint32_t erode(uint8_t neighboursToSurvive, bool inPlace, std::vector<Point<uint32_t>>* d = nullptr);

    // Grows the image inside the ROI at the current slice
    // Returns the number of grown pixels
    // neighboursToRevive: A pixel is grown if the number of neighbours is equal or greater
    // inPlace: if true, one revived pixel will affect the revival of the next one
    uint32_t grow(uint8_t neighboursToRevive, bool inPlace);
    
private:
    T* mp_data = nullptr;
// ========== CONSTRUCTORS ==========
    ImageHandler2D(); // No publically available default ctor
    ImageHandler2D(
        std::string name,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        uint8_t atype,
        uint8_t numKeywords,
        uint32_t circBufSize = 10);
    ImageHandler2D(std::string imName, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ);

// ========== HELPER FUNCTIONS ==========
    
    T cvtElmt(void* ptr, uint8_t atype, uint32_t index)
    {
        return convertAtypeArrayElement<T>(ptr, atype, index);
    }
};

// ######################################
// Implementation
// ######################################

// ===== specifications of factory functions =====

// Default implementation, throws an error
template <typename T>
inline spImHandler2D(T) ImageHandler2D<T>::newImageHandler2D(
    std::string name,
    size_t width,
    size_t height,
    size_t depth,
    uint8_t numKeywords,
    uint32_t circBufSize)
{
    throw std::runtime_error(
        "ImageHandler2D<T>::newImageHandler2D: Type not supported.");
    return nullptr;
}

// Defining specific factory functions via a makro ... way shorter!
#define IH_FACTORY(type, atype)                                         \
template <>                                                             \
inline spImHandler2D(type) ImageHandler2D<type>::newImageHandler2D(        \
    std::string name,                                                   \
    size_t width,                                                       \
    size_t height,                                                      \
    size_t depth,                                                       \
    uint8_t numKeywords,                                                \
    uint32_t circBufSize)                                               \
{                                                                       \
    spImHandler2D(type) sp(new ImageHandler2D<type>(                     \
        name,                                                           \
        width,                                                          \
        height,                                                         \
        depth,                                                          \
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
inline spImHandler2D(T) ImageHandler2D<T>::newHandler2DfrmImage(
        std::string name,
        IMAGE* im,
        uint8_t numKeywords,
        uint32_t circBufSize)
{
    std::vector<uint32_t> sVec = imSizeToVector(im);
    spImHandler2D(T) spIH = 
        newImageHandler2D(name, sVec[0], sVec[1], sVec[2], numKeywords, circBufSize);
    spIH->cpy(im);
    return spIH;
}

template <typename T>
inline spImHandler2D(T) ImageHandler2D<T>::newHandler2DAdoptImage(std::string imName)
{
    IMAGE im;
    errno_t err = ImageStreamIO_openIm(&im, imName.c_str());
    if (err != 0)
        throw std::runtime_error("ImageHandler2D::newHandlerAdoptImage: could not open image.");

    // Check if the call has been done correctly
    uint8_t atypeIm = im.md->datatype;
    uint8_t atypeArg = getAtype<T>();
    if (atypeIm != atypeArg)
        throw std::runtime_error("ImageHandler2D::newHandlerAdoptImage: image atype does not match the generic type of this call.");
    
    std::vector<uint32_t> sVec = imSizeToVector(&im);
    spImHandler2D(T) sp(new ImageHandler2D<T>(im.name, sVec[0], sVec[1], sVec[2]));

    ImageStreamIO_closeIm(&im);
    return sp;
}

// ===== specifications for image handling =====

template <typename T>
inline void ImageHandler2D<T>::cpy(IMAGE* im, bool update)
{   
    // Check image dimension
    if (mDepth == 1 && im->md->naxis > 2)
        throw std::runtime_error("ImageHandler2D::cpy: dimension of source image too high.\n");
    else if (mDepth > 1 && im->md->naxis != 3)
        throw std::runtime_error("ImageHandler2D::cpy: expected three dimensions.\n");
    
    // Check image sizes
    std::vector<uint32_t> sVec = imSizeToVector(im);
    if (sVec != getSizeVector())
        throw std::runtime_error("ImageHandler2D::cpy: Incompatible size of input image.\n");

    // Prepare read-buffer
    void* readBuffer;
    ImageStreamIO_readLastWroteBuffer(im, &readBuffer);

    // Convert
    uint8_t atype = im->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mp_data[i] = cvtElmt(readBuffer, atype, i);
    
    if (update)
        updateWrittenImage();
}

template <typename T>
inline void ImageHandler2D<T>::cpy_subtract(IMAGE* A, IMAGE* B, bool update)
{
    // Check image sizes
    std::vector<uint32_t> sVec = getSizeVector();
    if (sVec != imSizeToVector(A) || sVec != imSizeToVector(B))
        throw std::runtime_error("ImageHandler2D::cpy_subtract: Incompatible size of input image(s).\n");

    // Prepare buffers
    void* readBufferA;
    ImageStreamIO_readLastWroteBuffer(A, &readBufferA);
    void* readBufferB;
    ImageStreamIO_readLastWroteBuffer(B, &readBufferB);

    // Convert and subtract
    uint8_t atypeA = A->md->datatype;
    uint8_t atypeB = B->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mp_data[i] = cvtElmt(readBufferA, atypeA, i)
            - cvtElmt(readBufferB, atypeB, i);

    if (update)
        updateWrittenImage();
}

template <typename T>
inline void ImageHandler2D<T>::cpy_thresh(IMAGE* im, double thresh, bool update)
{
    // Check image sizes
    if (getSizeVector() != imSizeToVector(im))
        throw std::runtime_error("ImageHandler2D::cpy_thresh: Incompatible size of input image.\n");

    // Prepare buffer
    void* readBuffer;
    ImageStreamIO_readLastWroteBuffer(im, &readBuffer);

    // Convert
    uint8_t atype = im->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mp_data[i] = convertAtypeArrayElement<double>(readBuffer, atype, i) > thresh;
    
    if (update)
        updateWrittenImage();
}

template <typename T>
inline void ImageHandler2D<T>::cpy_convolve(IMAGE* A, IMAGE* K, bool update)
{
    // Check image size
    if (getSizeVector() != imSizeToVector(A))
        throw std::runtime_error("ImageHandler2D::cpy_convolve: Incompatible size of input A.\n");

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
    
    // Do convolution of all slices
    for (int32_t is = 0; is < mDepth; is++)
    {
        // Do convolution of slice
        for (int32_t ix = 0; ix < mWidth; ix++)
            for (int32_t iy = 0; iy < mHeight; iy++)
            {
                r = 0;
                for (int32_t kx = -kcX; kx < kw-kcX; kx++)
                    for (int32_t ky = -kcY; ky < kh-kcY; ky++)
                    {
                        if (ix+kx >= 0 && ix+kx < mWidth                            // Only inside of A
                            && iy+ky >= 0 && iy+ky < mHeight)
                        {
                            bi = (ky+kcY)*kw + kx+kcX;                              // Kernel index
                            k = convertAtypeArrayElement<float>(bK, tK, bi);        // Kernel value
                            bi = is*mWidth*mHeight + (iy+ky)*mWidth + ix+kx;        // Image index
                            r += convertAtypeArrayElement<float>(bA, tA, bi) * k;   // Convolution value
                        }
                    }
                mp_data[is*mWidth*mHeight + iy*mWidth + ix] = (T) r;
            }
    }
    
    if (update)
        updateWrittenImage();
}


// Implementation of ctor
template<typename T>
inline ImageHandler2D<T>::ImageHandler2D(
        std::string name,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        uint8_t atype,
        uint8_t numKeywords,
        uint32_t circBufSize)
        :
        ImageHandler2DBase(width, height, depth)
{
    int naxis = mDepth > 1 ? 3 : 2;
    uint32_t * imsize = new uint32_t[naxis]();
    imsize[0] = mWidth;
    imsize[1] = mHeight;
    if (mDepth > 1)
        imsize[2] = mDepth;
    int shared = 1; // image will be in shared memory
    ImageStreamIO_createIm_gpu(
            mp_image,
            name.c_str(),
            naxis,
            imsize,
            atype,
            -1,                 // -1: CPU RAM, 0+ : GPU
            shared,             // shared?
            10,                 // # of semaphores
            numKeywords,        // # of keywords
            0,                  // Imagetype - unknown
            circBufSize // circular buffer size (if shared), 0 if not used
        );
    delete imsize;
    updateImMetadata();
    mp_data = (T*) mp_h_imData;
}

// Implementation of adoption ctor
template<typename T>
inline ImageHandler2D<T>::ImageHandler2D(std::string imName, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ)
    : ImageHandler2DBase(sizeX, sizeY, sizeZ)
{
    // Open the image
    ImageStreamIO_openIm(mp_image, imName.c_str());
    ImageStreamIO_writeBuffer(mp_image, (void**) &mp_data);
    // Update the metadata of the parent class
    updateImMetadata();
    // Make the image persistent as default
    setPersistent(true);
}

template <typename T>
uint32_t ImageHandler2D<T>::erode(uint8_t neighboursToSurvive, bool inPlace, std::vector<Point<uint32_t>>* d)
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

    T* readBuffer;
    if (inPlace)
    {   // Read buffer equals write buffer.
        // Eroded pixels affect next pixel.
        readBuffer = mp_data;
    }
    else
    {   // Make a copy of the pixel data to read unchanged data
        readBuffer = new T[mNumPx];
        for (int i = 0; i < mNumPx; i++)
            readBuffer[i] = mp_data[i];
    }

    // Erode the ROI
    for (uint32_t ix = 0; ix < mROI.w(); ix++)
        for (uint32_t iy = 0; iy < mROI.h(); iy++)
        {   
            x = fromROIxToImX(ix);
            y = fromROIyToImY(iy);
            // Only consider pixel if it is greater than 0
            if (mp_data[m_currentSlice*mWidth*mHeight + y*mWidth + x] > 0)
            {   // Count the neighbours
                neighbours = 0;
                for (int j = 0; j < 8; j++)
                {
                    if (x+nOffX[j] >= 0 && x+nOffX[j] < mWidth &&
                        y+nOffY[j] >= 0 && y+nOffY[j] < mHeight)
                    {
                        neighbourIndex =
                            m_currentSlice*mWidth*mHeight + (y+nOffY[j])*mWidth + x + nOffX[j];
                        if (readBuffer[neighbourIndex] > 0)
                            neighbours++;
                    }
                }
                if (neighbours == 0)
                {   // Standalonepixel - Dissolving particle.
                    if (d != nullptr)
                        d->push_back(Point<uint32_t>(
                                x,y,mp_data[m_currentSlice*mWidth*mHeight + y*mWidth + x], true));
                    // Set to zero.
                    mp_data[m_currentSlice*mWidth*mHeight + y*mWidth + x] = 0;
                }
                else if (neighbours < neighboursToSurvive)
                    // Edgepixel - set to zero.
                    mp_data[m_currentSlice*mWidth*mHeight + y*mWidth + x] = 0;
                else
                    // Embedded pixel, leave as it is.
                    remainingPixels++;
            }
        }
    
    if (!inPlace)
        delete[] readBuffer;
    
    updateWrittenImage();
    return remainingPixels;
}

template <typename T>
uint32_t ImageHandler2D<T>::grow(uint8_t neighboursToRevive, bool inPlace)
{
    uint32_t revivedPixels = 0;
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

    T* readBuffer;
    if (inPlace)
    {   // Read buffer equals write buffer.
        // Eroded pixels affect next pixel.
        readBuffer = mp_data;
    }
    else
    {   // Make a copy of the pixel data to read unchanged data
        readBuffer = new T[mNumPx];
        for (int i = 0; i < mNumPx; i++)
            readBuffer[i] = mp_data[i];
    }

    // Grow the ROI
    for (uint32_t ix = 0; ix < mROI.w(); ix++)
        for (uint32_t iy = 0; iy < mROI.h(); iy++)
        {   
            x = fromROIxToImX(ix);
            y = fromROIyToImY(iy);
            // Only consider pixel if it is less than 1
            if (mp_data[m_currentSlice*mWidth*mHeight + y*mWidth + x] < 1)
            {   // Count the neighbours
                neighbours = 0;
                for (int j = 0; j < 8; j++)
                {
                    if (x+nOffX[j] >= 0 && x+nOffX[j] < mWidth &&
                        y+nOffY[j] >= 0 && y+nOffY[j] < mHeight)
                    {
                        neighbourIndex =
                            m_currentSlice*mWidth*mHeight + (y+nOffY[j])*mWidth + x + nOffX[j];
                        if (readBuffer[neighbourIndex] > 0)
                            neighbours++;
                    }
                }
                if (neighbours >= neighboursToRevive)
                {   // Edgepixel - set to zero.
                    mp_data[m_currentSlice*mWidth*mHeight + y*mWidth + x] = 1;
                    revivedPixels++;
                }
            }
        }
    
    if (!inPlace)
        delete[] readBuffer;
    
    updateWrittenImage();
    return revivedPixels;
}

#endif // IMAGEHANDLER_HPP