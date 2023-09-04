#include "wfgrad.hpp"
#include <iostream>
#include <cmath>

spWFGrad WFGrad::makeWFGrad(spPupil pupil)
{
    return spWFGrad(new WFGrad(pupil));
}
WFGrad::~WFGrad()
{
    delete[] mData;
}

double* WFGrad::getDataPtr(int* arrSizeOut)
{
    *arrSizeOut = mPupil->getNumValidFields()*2;
    return mData;
}

double* WFGrad::getDataPtrDX(int* arrSizeOut)
{
    *arrSizeOut = mPupil->getNumValidFields();
    return ptrDX;
}

double* WFGrad::getDataPtrDY(int* arrSizeOut)
{
    *arrSizeOut = mPupil->getNumValidFields();
    return ptrDY;
}

void WFGrad::printGrd()
{
    double* grdX2D = mPupil->createNew2DarrFromValues(
        mPupil->getNumValidFields(), ptrDX, (double) NAN);
    double* grdY2D = mPupil->createNew2DarrFromValues(
        mPupil->getNumValidFields(), ptrDY, (double) NAN);

    int w = mPupil->getWidth();
    printf("DX:\n");
    for (int y = 0; y < mPupil->getHeight(); y++)
        for (int x = 0; x < w; x++)
        {
            if (std::isnan(grdX2D[y*w+x]))
                printf("NaN\t");
            else
                printf("%.2f\t", grdX2D[y*w+x]);
            if (x == w-1)
                printf("\n");
        }
    printf("DY:\n");
    for (int y = 0; y < mPupil->getHeight(); y++)
        for (int x = 0; x < w; x++)
        {
            if (std::isnan(grdY2D[y*w+x]))
                printf("NaN ");
            else
                printf("%.2f\t", grdY2D[y*w+x]);
            if (x == w-1)
                printf("\n");
        }

    delete[] grdX2D;
    delete[] grdY2D;
}

double WFGrad::scalarProduct(spWFGrad other)
{
    int otherSize;
    double* otherData = other->getDataPtr(&otherSize);
    if (otherSize != mPupil->getNumValidFields()*2)
        return NAN;
    
    double scalarProduct = 0;
    for (int i = 0; i < otherSize; i++)
        scalarProduct += mData[i]*otherData[i];
    return scalarProduct;
}

WFGrad::WFGrad(spPupil pupil)
    : mPupil(pupil)
{
    mData = new double[mPupil->getNumValidFields()*2];
    ptrDX = mData;
    ptrDY = mData+mPupil->getNumValidFields();
}
