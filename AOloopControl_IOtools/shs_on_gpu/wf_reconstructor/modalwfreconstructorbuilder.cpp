#include "modalwfreconstructorbuilder.hpp"

ModalWFReconstructorBuilder::ModalWFReconstructorBuilder(
        spImageHandler(float) mask,
        std::string streamPrefix,
        int numModes)
    : mStreamPrefix(streamPrefix)
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
    mModeGenerator = WFGradModeGenerator::makeWFGradModeGenerator(samples, numModes);
    
    calcModes(numModes);
}

void ModalWFReconstructorBuilder::calcModes(int numModes)
{
    auto modes = mModeGenerator->calculateModes(numModes);
    mReconstructor = ModalWFReconstructor::makeWFReconstructor(modes, mStreamPrefix);
}

void ModalWFReconstructorBuilder::printTest()
{
    printf("\n\n\nSAMPLE TEST\n\n\n");
    std::pair<spWF, spWFGrad> testSample = mModeGenerator->getWFSample(mModeGenerator->getNumWFSamples()*0.3);

    int gradSize;
    double* gradientDbl = testSample.second->getDataPtr(&gradSize);
    float gradientFlt[gradSize];

    int wfSize;
    double* wfDbl = testSample.first->getDataPtr(&wfSize);
    float wfFlt[wfSize];

    for (int i = 0; i < mPupil->getNumValidFields()*2; i++)
        gradientFlt[i] = (float) gradientDbl[i];
    
    mReconstructor->reconstructWavefrontArrGPU_h2h(gradSize, gradientFlt, wfSize, wfFlt);

    printf("\nRef:\n");
    testSample.first->printWF();
    printf("\nReconst:\n");
    float* wf2D = mPupil->createNew2DarrFromValues(mWfLenght, wfFlt, NAN);
    int w = mPupil->getWidth();
    for (int y = 0; y < mPupil->getHeight(); y++)
        for (int x = 0; x < w; x++)
        {
            printf("%.6f\t", wf2D[y*w+x]);
            if (x == w-1)
                printf("\n");
        }

    printf("\n\n\nTILT TEST\n\n\n");
    for (int i = 0; i < gradSize; i++)
        gradientFlt[i] = i < wfSize ? 0 : 1; // x-gradient = 0, y-gradient = 1
    mReconstructor->reconstructWavefrontArrGPU_h2h(gradSize, gradientFlt, wfSize, wfFlt);
    printf("\nWF:\n");
    mPupil->fill2DarrWithValues(mWfLenght, wfFlt, mPupil->get2DarraySize(), wf2D, NAN);
    for (int y = 0; y < mPupil->getHeight(); y++)
        for (int x = 0; x < w; x++)
        {
            printf("%.6f\t", wf2D[y*w+x]);
            if (x == w-1)
                printf("\n");
        }

    delete[] wf2D;
}