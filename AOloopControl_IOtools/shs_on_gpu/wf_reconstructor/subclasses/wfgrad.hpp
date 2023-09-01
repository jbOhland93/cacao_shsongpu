#ifndef WFGRAD_HPP
#define WFGRAD_HPP
#include "pupil.hpp"

#define spWFGrad std::shared_ptr<WFGrad>

// A class representing wavefront gradients
class WFGrad {
public:
    static spWFGrad makeWFGrad(spPupil pupil);
    ~WFGrad();

    spPupil getPupil() { return mPupil; }
    double* getDataPtr(int* arrSizeOut);
    double* getDataPtrDX(int* arrSizeOut);
    double* getDataPtrDY(int* arrSizeOut);

    void printGrd();
    double scalarProduct(spWFGrad other);
    
private:
    spPupil mPupil;
    double* mData; // The gradient values within the pupil, stored as 1D array
    double* ptrDX; // The starting point of the gradient in x-direction within mData
    double* ptrDY; // The starting point of the gradient in y-direction within mData

    WFGrad(); // No publically available Ctor
    WFGrad(spPupil pupil);
};

#endif // WFGRAD_HPP