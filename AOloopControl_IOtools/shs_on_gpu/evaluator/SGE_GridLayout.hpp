#ifndef SGE_GRIDLAYOUT_HPP
#define SGE_GRIDLAYOUT_HPP

#include "../util/CudaSGEutil.hpp"
#include "SGE_ReferenceManager.hpp"

#define spGridLayout std::shared_ptr<SGE_GridLayout>
class SGE_GridLayout
{
public:
    // Factory function
    static spGridLayout makeGridLayout(int device, spRefManager refManager);
    // Dtor
    ~SGE_GridLayout();

    // Returns the number of coordinates per subapertur for which a convolution
    // with a gaussian kernel shall be calculated.
    static int getNumConvolutionsPerAp() { return 12; }

    // Public fields which will be used the CUDA kernels
    // GPU data
    int mDeviceID;              // Device ID
    int mNumCudaCores;          // Cuda cores on GPU
    int mWarpSize;              // GPU warp size

    // Grid dimensioning
    int mWarpsPerBlock;         // Warps per block
    int mBlockSize;             // Threads per block
    int mNumSubapertures;       // Total # of apertures
    int mNumBlocks;             // # of blocks in grid

    // Aperture data
    int mWindowSize;            // Edge size of considered window per subaperture
    int mNumWindowPx;           // # of pixels inside one aperture window
    int mKernelSize;            // Edge size of the convolution kernel
    int mNumKernelPx;           // # of pixels in the kernel
    int mCorrelMargin;          // Margin in px between window edge and convolution area
    int mNumCorrelPosPerAp      // # of positions for which the convolution shall be calced
        = getNumConvolutionsPerAp();

    // Shared memory organization
    int mShmSize;               // Size of the dynamic shared memory per block in bytes
    int mShmImDataOffset;       // Offset to the image data buffer in dynamic shm
    int mShmImDataSize;         // Size of the image data buffer in dynamic shm
    int mShmKernelOffset;       // Offset to the kernel buffer in dynamic shm
    int mShmKernelSize;         // Size of the kernel buffer in dynamic shm
    int mShmConvCoordsXOffset;  // Offset to the X buffer of the coordinates to be convoluted in dynamic shm
    int mShmConvCoordsXSize;    // Size of the X buffer of the coordinates to be convoluted in dynamic shm
    int mShmConvCoordsYOffset;  // Offset to the Y buffer of the coordinates to be convoluted in dynamic shm
    int mShmConvCoordsYSize;    // Size of the Y buffer of the coordinates to be convoluted in dynamic shm
    int mShmConvBuf1Offset;     // Offset to the 1st buffer for the convolution operation in dynamic shm
    int mShmConvBuf1Size;       // Size of the 1st buffer for the convolution operation in dynamic shm
    int mShmConvBuf2Offset;     // Offset to the 2nd buffer for the convolution operation in dynamic shm
    int mShmConvBuf2Size;       // Size of the 2nd buffer for the convolution operation in dynamic shm
    int mShmConvResultOffset;   // Offset to the convolution result buffer in dynamic shm
    int mShmConvResultSize;     // Size of the convolution result buffer in dynamic shm

    // Correlation offsets relative to the window root.
    // Array length equals the return value of getNumConvolutionsPerAp().
    // Stored on the device as long as this object is not destroyed.
    int* mp_d_CorrelationOffsetsX = nullptr;
    int* mp_d_CorrelationOffsetsY = nullptr;

    // Returns a pointer to the copy of this object in the global device memory
    SGE_GridLayout* getDeviceCopy() { return mpd_deviceCopy; }

private:
    SGE_GridLayout(); // No publically available default ctor.
    // Ctor
    SGE_GridLayout(int device, spRefManager refManager);
    
    // Reads relevant properties about the used GPU and stores them in member fields
    void getGPUproperties();
    // Helper function, looks up the cuda cores based on the compute capability
    int getCudaCoresPerSM(int major, int minor);
    // Calculates the grid porpiertes from what is known
    void calcGridProperties();
    // Sets the sizes and offsets for the dynamic shared memory for ach block on the grid
    void setSHMlayout();
    // Allocates and writes coordinate offsets relative to the root
    // of a subaperture to the given locations.
    void writeCorrelOffsetsFromWindowRootToDevice();
    // Prints a little report about the properties of the layout
    void printReport();

    SGE_GridLayout* mpd_deviceCopy;
};

#endif // SGE_GRIDLAYOUT_HPP