#ifndef PUPIL_HPP
#define PUPIL_HPP
#include <memory>

#define spPupil std::shared_ptr<Pupil>

class Pupil
{
public:
    static spPupil makePupil(int width, int height, uint8_t* pupilArr);
    ~Pupil();

    int getWidth() { return mWidth; }
    int getHeight() { return mHeight; }
    int get2DarraySize() { return mWidth*mHeight; }
    int getNumValidFields() { return mNumValidFields; }
    bool* getDataPtr(int* arrSizeOut);

    void printPupil();

    template <typename T>
    void fill2DarrWithValues(T* values, T* dst, T invalidValue);

    bool isInProximity(int pX, int pY, double distance);

private:
    int mWidth;
    int mHeight;
    int mNumValidFields;
    bool* mPupilArr = nullptr;

    Pupil(); // No publically available Ctor
    Pupil(int width, int height, uint8_t* pupilArr);
};

template <typename T>
void Pupil::fill2DarrWithValues(T* values, T* dst, T invalidValue)
{
    int i = 0;
    for (int y = 0; y < mHeight; ++y) {
        for (int x = 0; x < mWidth; ++x) {
            if (mPupilArr[y*mWidth+x])
            {
                dst[y*mWidth+x] = values[i];
                i++;
            }
            else
                dst[y*mWidth+x] = invalidValue;
        }
    }
}

#endif // PUPIL_HPP
