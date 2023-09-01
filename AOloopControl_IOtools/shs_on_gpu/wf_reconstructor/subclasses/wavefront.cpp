#include "wavefront.hpp"
#include <iostream>
#include <cmath>

spWF Wavefront::makeWavefront(spPupil pupil)
{
    return spWF(new Wavefront(pupil));
}
Wavefront::~Wavefront()
{
    delete[] mData;
}

double* Wavefront::getDataPtr(int* arrSizeOut)
{
    *arrSizeOut = mPupil->getNumValidFields();
    return mData;
}

void Wavefront::subtractMean()
{
    // Calculate the mean value
    double sum;
    for (int i = 0; i < mPupil->getNumValidFields(); i++)
        sum += mData[i];
    double mean = sum / mPupil->getNumValidFields();

    // Subtract the mean from all the samples
    for (int i = 0; i < mPupil->getNumValidFields(); i++)
        mData[i] -= mean;
}

void Wavefront::printWF()
{
    double* wf2D = new double[mPupil->get2DarraySize()];
    mPupil->fill2DarrWithValues(mData, wf2D, (double) NAN);

    int w = mPupil->getWidth();
    for (int y = 0; y < mPupil->getHeight(); y++)
        for (int x = 0; x < w; x++)
        {
            printf("%.2f\t", wf2D[y*w+x]);
            if (x == w-1)
                printf("\n");
        }

    delete[] wf2D;
}

double Wavefront::scalarProduct(spWF other)
{
    int otherSize;
    double* otherData = other->getDataPtr(&otherSize);
    if (otherSize != mPupil->getNumValidFields())
        return NAN;
    
    double scalarProduct = 0;
    for (int i = 0; i < otherSize; i++)
        scalarProduct += mData[i]*otherData[i];
    return scalarProduct;
}

Wavefront::Wavefront(spPupil pupil)
    : mPupil(pupil)
{
    mData = new double[mPupil->getNumValidFields()];
}
