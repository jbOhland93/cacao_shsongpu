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

    // Allocates an array with a size of get2DarraySize()
    template <typename T>
    T* allocate2Darr();
    // Fills a pupil array with values,
    // where each value goes into the next valid field
    // The lengths are used as a sanity check.
    template <typename T>
    void fill2DarrWithValues(int valuesLen, T* values, int dstLen, T* dst, T invalidValue);
    template <typename T>
    T* createNew2DarrFromValues(int valuesLen, T* values, T invalidValue);

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
void Pupil::fill2DarrWithValues(
    int valuesLen, T* values, int dstLen, T* dst, T invalidValue)
{
    if (valuesLen != mNumValidFields)
        throw std::runtime_error("Pupil::fill2DarrWithValues: values length does not match getNumValidFields().");
    if (dstLen != get2DarraySize())
        throw std::runtime_error("Pupil::fill2DarrWithValues: destination length does not match get2DarraySize().");
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

template <typename T>
T* Pupil::allocate2Darr()
{
    return new T[get2DarraySize()];
}

template <typename T>
T* Pupil::createNew2DarrFromValues(int valuesLen, T* values, T invalidValue)
{
    T* dst = allocate2Darr<T>();
    fill2DarrWithValues(valuesLen, values, get2DarraySize(), dst, invalidValue);
    return dst;
}

#endif // PUPIL_HPP
