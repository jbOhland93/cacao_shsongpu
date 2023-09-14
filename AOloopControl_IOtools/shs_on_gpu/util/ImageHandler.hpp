// This is a generic image handler, i.e. it should work for all datatypes.
// Generic specifications cannot be written into the compile unit (.cpp)
// Therefore, this is a long and inline heavy file. Sorry for that.

#ifndef IMAGEHANDLER_HPP
#define IMAGEHANDLER_HPP

#include "ImageHandlerBase.hpp"
#include "atypeUtil.hpp"
#include <memory>
#include <vector>
#include <limits>

#define spImageHandler(type) std::shared_ptr<ImageHandler<type>>
#define newImHandlerFrmIm(type, name, image) ImageHandler<type>::newHandlerfrmImage(name, image)

template <typename T>
class ImageHandler : public ImageHandlerBase
{
public:

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
    // Creates an image handler that operates on an already existing image.
    // Throws an error if the generic type does not match the image type.
    static spImageHandler(T) newHandlerAdoptImage(std::string imName);

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
        { return mp_data[fromROIyToImY(y)*mWidth + fromROIxToImX(x)]; }
    // Writes the given element at teh x/y position into the write buffer
    void write(T e, uint32_t x, uint32_t y)
        { mp_data[fromROIyToImY(y)*mWidth + fromROIxToImX(x)] = e; }
    // Reads all the samples at x/y from the circular buffer
    std::vector<T> readCircularBufAt(uint32_t x, uint32_t y)
    {
        std::vector<T> v;
        T* cbBuf = (T*) mp_image->CBimdata;
        for (int i = 0; i < mp_image->md->CBsize; i++)
            v.push_back(cbBuf[i*mNumPx + y*mWidth + x]);
        return v;
    }
    

// ========== OPERATIONS ==========
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
    // neighboursToSurvive: selfexplanatory
    // inPlace: if true, one eroded pixel will affect the survival of the next one
    // d: the coordinates to dissolved particles are appended to this vector
    uint32_t erode(uint8_t neighboursToSurvive, bool inPlace, std::vector<Point<uint32_t>>* d = nullptr);
    
private:
    T* mp_data = nullptr;
// ========== CONSTRUCTORS ==========
    ImageHandler(); // No publically available default ctor
    ImageHandler(
        std::string name,
        uint32_t width,
        uint32_t height,
        uint8_t atype,
        uint8_t numKeywords,
        uint32_t circBufSize = 10);
    ImageHandler(std::string imName, uint32_t sizeX, uint32_t sizeY);

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
inline spImageHandler(T) ImageHandler<T>::newImageHandler(
    std::string name,
    size_t width,
    size_t height,
    uint8_t numKeywords,
    uint32_t circBufSize)
{
    throw std::runtime_error(
        "ImageHandler<T>::newImageHandler: Type not supported.");
    return nullptr;
}

// Defining specific factory functions via a makro ... way shorter!
#define IH_FACTORY(type, atype)                                         \
template <>                                                             \
inline spImageHandler(type) ImageHandler<type>::newImageHandler(    \
    std::string name,                                                   \
    size_t width,                                                       \
    size_t height,                                                      \
    uint8_t numKeywords,                                                \
    uint32_t circBufSize)                                               \
{                                                                       \
    spImageHandler(type) sp(new ImageHandler<type>(                 \
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
inline spImageHandler(T) ImageHandler<T>::newHandlerfrmImage(
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

template <typename T>
inline spImageHandler(T) ImageHandler<T>::newHandlerAdoptImage(std::string imName)
{
    IMAGE im;
    errno_t err = ImageStreamIO_openIm(&im, imName.c_str());
    if (err != 0)
        throw std::runtime_error("ImageHandler::newHandlerAdoptImage: could not open image.");

    // Check if the call has been done correctly
    uint8_t atypeIm = im.md->datatype;
    uint8_t atypeArg = getAtype<T>();
    if (atypeIm != atypeArg)
        throw std::runtime_error("ImageHandler::newHandlerAdoptImage: image atype does not match the generic type of this call.");
    
    spImageHandler(T) sp(new ImageHandler<T>(im.name, im.md->size[0], im.md->size[1]));

    ImageStreamIO_closeIm(&im);
    return sp;
}

// ===== specifications for image handling =====

template <typename T>
inline void ImageHandler<T>::cpy(IMAGE* im)
{
    // Check image sizes
    uint32_t* size = im->md->size;
    if (size[0] != mWidth || size[1] != mHeight)
        throw std::runtime_error("ImageHandler::cpy: Incompatible size of input image.\n");

    // Prepare read-buffer
    void* readBuffer;
    ImageStreamIO_readLastWroteBuffer(im, &readBuffer);

    // Convert
    uint8_t atype = im->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mp_data[i] = cvtElmt(readBuffer, atype, i);
    
    updateWrittenImage();
}

template <typename T>
inline void ImageHandler<T>::cpy_subtract(IMAGE* A, IMAGE* B)
{
    // Check image sizes
    uint32_t* sizeA = A->md->size;
    uint32_t* sizeB = B->md->size;
    if (   sizeA[0] != mWidth
        || sizeB[0] != mWidth
        || sizeA[1] != mHeight
        || sizeB[1] != mHeight)
        throw std::runtime_error("ImageHandler::cpy_subtract: Incompatible size of input image(s).\n");

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

    updateWrittenImage();
}

template <typename T>
inline void ImageHandler<T>::cpy_thresh(IMAGE* im, double thresh)
{
    // Check image sizes
    uint32_t* size = im->md->size;
    if (size[0] != mWidth || size[1] != mHeight)
        throw std::runtime_error("ImageHandler::cpy_thresh: Incompatible size of input image.\n");

    // Prepare buffer
    void* readBuffer;
    ImageStreamIO_readLastWroteBuffer(im, &readBuffer);

    // Convert
    uint8_t atype = im->md->datatype;
    for (int i = 0; i < mNumPx; i++)
        mp_data[i] = convertAtypeArrayElement<double>(readBuffer, atype, i) > thresh;
    
    updateWrittenImage();
}

template <typename T>
inline void ImageHandler<T>::cpy_convolve(IMAGE* A, IMAGE* K)
{
    // Check image size
    uint32_t* size = A->md->size;
    if (size[0] != mWidth || size[1] != mHeight)
        throw std::runtime_error("ImageHandler::cpy_convolve: Incompatible size of input A.\n");

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
                    if (ix+kx >= 0 && ix+kx < mWidth                            // Only inside of A
                        && iy+ky >= 0 && iy+ky < mHeight)
                    {
                        bi = (ky+kcY)*kw + kx+kcX;                              // Kernel index
                        k = convertAtypeArrayElement<float>(bK, tK, bi);        // Kernel value
                        bi = (iy+ky)*mWidth + ix+kx;                            // Image index
                        r += convertAtypeArrayElement<float>(bA, tA, bi) * k;   // Convolution value
                    }
                }
            mp_data[iy*mWidth + ix] = (T) r;
        }
    
    updateWrittenImage();
}


// Implementation of ctor
template<typename T>
inline ImageHandler<T>::ImageHandler(
        std::string name,
        uint32_t width,
        uint32_t height,
        uint8_t atype,
        uint8_t numKeywords,
        uint32_t circBufSize)
        :
        ImageHandlerBase(width, height)
{
    int naxis = 2;
    uint32_t * imsize = new uint32_t[naxis]();
    imsize[0] = mWidth;
    imsize[1] = mHeight;
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
inline ImageHandler<T>::ImageHandler(std::string imName, uint32_t sizeX, uint32_t sizeY)
    : ImageHandlerBase(sizeX, sizeY)
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
uint32_t ImageHandler<T>::erode(uint8_t neighboursToSurvive, bool inPlace, std::vector<Point<uint32_t>>* d)
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
            if (mp_data[y*mWidth + x] > 0)
            {   // Count the neighbours
                neighbours = 0;
                for (int j = 0; j < 8; j++)
                {
                    if (x+nOffX[j] >= 0 && x+nOffX[j] < mWidth &&
                        y+nOffY[j] >= 0 && y+nOffY[j] < mHeight)
                    {
                        neighbourIndex = (y+nOffY[j])*mWidth + x + nOffX[j];
                        if (readBuffer[neighbourIndex] > 0)
                            neighbours++;
                    }
                }
                if (neighbours == 0)
                {   // Standalonepixel - Dissolving particle.
                    if (d != nullptr)
                        d->push_back(
                            Point<uint32_t>(x,y,mp_data[y*mWidth + x], true));
                    // Set to zero.
                    mp_data[y*mWidth + x] = 0;
                }
                else if (neighbours < neighboursToSurvive)
                    // Edgepixel - set to zero.
                    mp_data[y*mWidth + x] = 0;
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

#endif // IMAGEHANDLER_HPP