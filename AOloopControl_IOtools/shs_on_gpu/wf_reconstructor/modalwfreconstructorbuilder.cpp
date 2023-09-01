#include "modalwfreconstructorbuilder.hpp"

ModalWFReconstructorBuilder::ModalWFReconstructorBuilder(spImageHandler(float) mask, int numModes)
{
    int pupilWidth = mask->mWidth;
    int pupilHeight = mask->mHeight;

    printf("TODO ModalWFReconstructorBuilder::ModalWFReconstructorBuilder: Replace pupil array type with uint8_t.\n");
    float* pupilNumericF = mask->getWriteBuffer();
    uint8_t pupilNumeric[pupilWidth*pupilHeight];
    for (int i = 0; i < pupilWidth*pupilHeight; i++)
        pupilNumeric[i] = (uint8_t) pupilNumericF[i];

    mPupil = Pupil::makePupil(pupilWidth, pupilHeight, pupilNumeric);
    mWfLenght = mPupil->getNumValidFields();
    mGradLength = mWfLenght*2;
    spRspnsSampler sampler = ResponseSampler::makeSampler(mPupil, 2.1);
    auto samples = sampler->generateSamples(2./(pupilWidth+pupilHeight));
    mModeGenerator = WFGradModeGenerator::makeWFGradModeGenerator(samples);
    
    calcModes(numModes);
}

ModalWFReconstructorBuilder::ModalWFReconstructorBuilder(int pupilWidth, int pupilHeight, uint8_t* pupilArr, int numModes)
{
    mPupil = Pupil::makePupil(pupilWidth, pupilHeight, pupilArr);
    mWfLenght = mPupil->getNumValidFields();
    mGradLength = mWfLenght*2;
    spRspnsSampler sampler = ResponseSampler::makeSampler(mPupil, 2.1);
    auto samples = sampler->generateSamples(2./(pupilWidth+pupilHeight));
    mModeGenerator = WFGradModeGenerator::makeWFGradModeGenerator(samples);
    
    calcModes(numModes);
}

void ModalWFReconstructorBuilder::calcModes(int numModes)
{
    auto modes = mModeGenerator->calculateModes(numModes);
    mReconstructor = ModalWFReconstructor::makeWFReconstructor(modes);
}

void ModalWFReconstructorBuilder::printTest()
{
    printf("\n\n\nSAMPLE TEST\n\n\n");
    std::pair<spWF, spWFGrad> testSample = mModeGenerator->getWFSample(mModeGenerator->getNumWFSamples()*0.3);
    spWF reconstructed = mReconstructor->reconstructWavefront(testSample.second);
    printf("\nRef:\n");
    testSample.first->printWF();
    printf("\nReconst:\n");
    reconstructed->printWF();

    printf("\n\n\nTILT TEST\n\n\n");
    spWFGrad grd = WFGrad::makeWFGrad(mPupil);
    int grdSize;
    double* grdPtr = grd->getDataPtrDX(&grdSize);
    for (int i = 0; i < grdSize; i++)
        grdPtr[i] = 0;
    grdPtr = grd->getDataPtrDY(&grdSize);
    for (int i = 0; i < grdSize; i++)
        grdPtr[i] = 1;
    reconstructed = mReconstructor->reconstructWavefront(grd);
    reconstructed->printWF();
}