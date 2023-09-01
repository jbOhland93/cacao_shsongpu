#include "pupil.hpp"
#include <cmath>
#include <iostream>

spPupil Pupil::makePupil(int width, int height, uint8_t* pupilArr)
{
    return spPupil(new Pupil(width, height, pupilArr));
}

Pupil::~Pupil()
{
    delete[] mPupilArr;
}

void Pupil::printPupil()
{
    for (int y = 0; y < mHeight; y++)
        for (int x = 0; x < mWidth; x++)
        {
            if (mPupilArr[y * mWidth + x])
                printf("1\t");
            else
                printf("0\t");
            if (x == mWidth - 1)
                printf("\n");
        }
}

bool* Pupil::getDataPtr(int* arrSizeOut)
{
    *arrSizeOut = get2DarraySize();
    return mPupilArr;
}

bool Pupil::isInProximity(int pX, int pY, double distance) {
    int xStart = std::max(0, pX - (int) ceil(distance));
    int yStart = std::max(0, pY - (int) ceil(distance));

    int xEnd = std::min(mWidth, pX + (int) ceil(distance));
    int yEnd = std::min(mHeight, pY + (int) ceil(distance));

    double d2 = std::pow(distance, 2);
    for (int y = yStart; y < yEnd; y++) {
        for (int x = xStart; x < xEnd; x++) {
            if (mPupilArr[y * mWidth + x]) {
                if (std::pow(x-pX, 2) + std::pow(y-pY, 2) < d2)
                    return true;
            }
        }
    }

    return false;
}

Pupil::Pupil(int width, int height, uint8_t* pupilArr)
    : mWidth(width), mHeight(height)
{
    mPupilArr = new bool[get2DarraySize()];

    mNumValidFields = 0;
    int minX = width;
    int maxX = 0;
    int minY = height;
    int maxY = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            mPupilArr[y*width+x] = pupilArr[y*width+x] != 0;
            if (mPupilArr[y*width+x]) {
                mNumValidFields++;
                if (minX > x)
                    minX = x;
                if (maxX < x)
                    maxX = x;
                if (minY > y)
                    minY = y;
                if (maxY < y)
                    maxY = y;
            }
        }
    }
}
