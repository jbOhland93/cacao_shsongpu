#ifndef MODALWFRECONSTRUCTORBUILDER_HPP
#define MODALWFRECONSTRUCTORBUILDER_HPP

#include "modalwfreconstructor.hpp"
#include "../util/ImageHandler.hpp"

// A class for building a modal WF reconstructor on a pupil
class ModalWFReconstructorBuilder {
public:
    ModalWFReconstructorBuilder(
        spImageHandler(uint8_t) mask,
        std::string streamPrefix,
        int numModes = -1);
    // (Re)calculates the reconstruction matrix, using the given number of WF modes.
    // If numModes is <= 0, all available modes (except piston) are used.
    void calcModes(int numModes = -1);

    // Returns the number of modes that are included in the current reconstruction matrix
    int getNumModes() { return mModeGenerator->getNumModes(); }
    // Returns the maximum number of modes that are available on the pupil, excluding piston
    int getMaxNumModes()  { return mModeGenerator->getMaxNumModes(); }

    // Returns the constructed, ready-to-use reconstructor object
    spWFReconst getReconstructor() { return mReconstructor; }

    // Prints some test arrays
    void printTest();
    
private:
    std::string mStreamPrefix;  // A prefix for the image streams
    spPupil mPupil;             // The pupil on which the reconstructor acts
    int mWfLenght;              // The length of a 1D wavefront array
    int mGradLength;            // The length of a 1D gradient array
    spWGModeGen mModeGenerator; // A generator object, calculating the modes on the pupil
    spWFReconst mReconstructor; // The WF reconstructor, managing the transfer matrix

    ModalWFReconstructorBuilder(); // No publically available standard Ctor
};


#endif //MODALWFRECONSTRUCTORBUILDER_HPP