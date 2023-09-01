#ifndef WAVEFRONT_HPP
#define WAVEFRONT_HPP
#include "pupil.hpp"

#define spWF std::shared_ptr<Wavefront>

// A class representing a wavefront
class Wavefront {
public:
    static spWF makeWavefront(spPupil pupil);
    ~Wavefront();

    spPupil getPupil() { return mPupil; }
    double* getDataPtr(int* arrSizeOut);
    void subtractMean();
    void printWF();
    double scalarProduct(spWF other);
    
private:
    spPupil mPupil;
    double* mData; // The wavefront values within the pupil, stored as 1D array

    Wavefront(); // No publically available Ctor
    Wavefront(spPupil pupil);
};

#endif // WAVEFRONT_HPP