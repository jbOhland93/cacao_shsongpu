#ifndef SPOTFITTER_HPP
#define SPOTFITTER_HPP

#include "../ref_recorder/SGR_ImageHandler.hpp"

#include "Rectangle.hpp"
#include "OneDGaussianFit.hpp"

// A class to determine the spotsize of a SHS image
class SpotFitter
{
public:
    SpotFitter(spImageHandler(float) imageHandler);
    
    // Sets a set of square ROIs for the fits.
    // ROIs which do not fully overlap with the image are discarded.
    template <typename T>
    void setROIs(std::vector<Point<T>> centers, uint32_t size);
    // Sets the lineout width in pixels.
    // This value will be used for the next call of "makeLineouts".
    // Musst be greater than 0 and smaller then the ROI size.
    void setLineoutWidth(uint32_t width);
    // Will generate lineouts for each ROI along the X and Y axis.
    // forceFullWidth: If false, mLineoutWidth will be used.
    // streamPrefix: Prefix for newly generated streams
    // visualize: If true, creates an image with the lineouts
    void makeLineouts(bool forceFullWidth = false,
        std::string streamPrefix = "",
        bool visualize = false);
    // Will fit gaussians along all pre-generated lineouts.
    // visualize: If true, creates images for the fit and the errors
    void fitCurves(std::string streamPrefix = "", bool visualize = false);
    // Returns the average standard deviation of the fit results
    // The averaging uses the fitted spot amplitude as weight.
    double getAvgStdDev();
    // Returns the average offset of the fit results
    double getAvgOffset();
    // Returns the centers of the fit results
    // The averaging uses the fitted spot amplitude as weight.
    std::vector<Point<double>> getFittedSpotCenters();

    // Performs spot fits to the given positions
    // centers: The estimated spot centers to be fitted
    // windowSize: The size around the spot centers
    // iterations: Each successive iteration optimizes the lineout
    //      width to the previously found spot size to optimize the SNR
    // streamPrefix: Prefix for newly generated streams
    // visualize: If true, images in SHM will be generated to monitor
    //      the accuracy of the final fit
    void expressFit(
        std::vector<Point<uint32_t>> centers,
        uint32_t windowSize,
        uint32_t iterations = 2,
        std::string streamPrefix = "",
        bool visualize = false);
    
private:
    const spImageHandler(float) mImageHandler;
    std::vector<Rectangle<uint32_t>> mSpotROIs;
    uint32_t mLineoutWidth = 0;
    uint32_t mLineoutLength = 0;
    std::vector<std::shared_ptr<double[]>> mLineoutsX;
    std::vector<std::shared_ptr<double[]>> mLineoutsY;
    std::vector<OneDGaussianFit::gaussianParams> mFitResultsX;
    std::vector<OneDGaussianFit::gaussianParams> mFitResultsY;

    SpotFitter(); // No public default constructor
    void printImStreamDescr(std::string imName, std::string contentShort);
};

// Default implementation
// Just convert the point vector and call the uint32_t implementation.
template<typename T>
inline void SpotFitter::setROIs(std::vector<Point<T>> centers, uint32_t size)
{
    std::vector<Point<uint32_t>> centersConverted;
    for (int i = 0; i < centers.size(); i++)
        centersConverted.push_back(centers.at(i));
    
    setROIs(centersConverted, size);
}

// uint32_t implementation
// This one is doing the actual work.
template<>
inline void SpotFitter::setROIs<uint32_t>(std::vector<Point<uint32_t>> centers, uint32_t size)
{
    mLineoutLength = size;
    // New ROIs means new coarse fit
    // Set the lineout width to the ROI size
    mLineoutWidth = size;
    // Calculate the roots of the ROI rects
    uint32_t roiOffset = floor(size/2.);
    // Re-populate the ROI vector
    mSpotROIs.clear();
    for (int i = 0; i < centers.size(); i++)
    {
        Point<uint32_t> root = centers.at(i) - roiOffset;
        Rectangle<uint32_t> roi(root, size, size);
        if (root.mX + size < mImageHandler->mWidth || // root.mx and root.my are >= 0 (unsigned)
            root.mY + size < mImageHandler->mHeight)
            mSpotROIs.push_back(roi);
    }

    // Clear the data vectors of subsequent steps
    mLineoutsX.clear();
    mLineoutsY.clear();
    mFitResultsX.clear();
    mFitResultsY.clear();
}

#endif // SPOTFITTER_HPP