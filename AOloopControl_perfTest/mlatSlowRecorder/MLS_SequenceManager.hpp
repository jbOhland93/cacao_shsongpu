#ifndef MLS_SEQUENCEMANAGER_HPP
#define MLS_SEQUENCEMANAGER_HPP

#include "../../AOloopControl_IOtools/shs_on_gpu/util/ImageHandler2D.hpp"
#include <chrono>

// Forward declaration
class MLS_ResultManager;

// A claas for recording and evaluating WFS sequences
// for latency recording of slow DMs
class MLS_SequenceManager
{
public:
    MLS_SequenceManager(
        FUNCTION_PARAMETER_STRUCT* fps, // process related FPS
        IMAGE* wfsStream,       // WFS output stream
        uint32_t numPokes,      // number of iterations
        uint32_t framesPerPoke, // number of frames per iteration
        bool saveRaw);          // Save raw data cubes

    // Measures the FPS over the given duration and returns the result
    double measureFPS(float measurementTime_s);

    // Record either the pre- or post-poke state
    // Both have to be done in two separate calls
    // The DM has to be poked and settled in between the calls
    void recordPokeResponse(bool postPoke);
    // After both DM states have been recorded, this function
    // evaluates the difference and stores a normalized response mode.
    void evalPokeResponse();

    // Set the response pattern instead of recording it
    void setPokeResponse(std::string respImName, uint32_t sliceIdx);

    // Record a sequence with a live DM poke.
    // If postPoke=false, the recording will stop after a frame
    // and return the exact time when the DM poke shall be issued.
    // After that, immediately call this function with postPoke=true
    // so no frame is missed.
    std::chrono::_V2::system_clock::time_point recordLatencyPokeSequence(
        uint32_t iteration,
        double interframePokeDelay_ns,
        bool postPoke);
    // Decomposes the last DM poke sequence into amplitudes.
    // Returns a pair where
    // - the first element is a ptr to the timestamps
    // - the second element is a ptr to the response amplitudes
    std::pair<double*, float*> decomposeLastPokeSequence(
        uint32_t iteration);

    // Calculate average raw 90%|110% settling latency
    double calcRawSettlingLatency();

    // Calculates a floating avg over all decomposed sequences.
    void smoothLatencyDecomposition(
            std::shared_ptr<MLS_ResultManager> p_resultMngr);

    // Calculates the delay between the poke command
    // and the first detected motion. Result is in us.
    // Uses the smoothed latency sequence decomposition.
    double calcHWdelay();

    // Calculates the DM rise time. Result is in us.
    // Uses the smoothed latency sequence decomposition.
    double calcRiseime();

    // Calculates the delay between the poke command
    // and the [90%|110%] settling time. Result is in us.
    // Uses the smoothed latency sequence decomposition.
    // Depending on the noise in the system, this could
    // therefore be an optimistic estimate.
    double calcHWlatency();

    // Waits until the WFS has delivered a given number of frames
    void waitNumFrames(uint32_t frames);

private:
    std::string m_streamPrefix = "mlat-slowDM_";
    FUNCTION_PARAMETER_STRUCT* mp_fps;
    uint32_t m_numPokes;
    uint32_t m_framesPerPoke;
    uint32_t m_framesPriorToPoke;
    bool m_saveRaw;

    std::string m_responseFilename = "PokeResponse";

    // == IMAGE HANDLERS ==
    // Adopted WFS output stream
    spImHandler2D(float) mp_IHwfs;
    // Sequence for measuring the poke respnse
    spImHandler2D(float) mp_IHpokeResponseSequence;
    // Single WF, containing the expected poke response
    spImHandler2D(float) mp_IHpokeResponse;
    // Sequence containing frames of one poke
    spImHandler2D(float) mp_IHLatencySequence;

    // Contains the relative timestamps of the pokes in ns
    // The corresponding amplitudes are stored in mp_IHpokeAmps
    // Format is double as the time is measured in ns since the last epoch
    // Here, float would not yield sufficient precision.
    spImHandler2D(double) mp_IHpokeTimesRel;
    // 3D image, containing the decomposed poke evolutions
    // The corresponding timestamps are stored in mp_IHpokeTimesRel
    spImHandler2D(float) mp_IHpokeAmps;
    
    // Relative times in us of the smoothed avg poke response amplitude
    // The corresponding amplitudes are stored in mp_IHpokeAmpSmoothed
    spImHandler2D(double) mp_IHpokeTimeSmoothed;
    // Smoothed avg poke response amplitude
    // The corresponding relative times are stored in mp_IHpokeTimeSmoothed
    spImHandler2D(float) mp_IHpokeAmpSmoothed;
    // Smoothed avg poke response standard deviation
    // The corresponding relative times are stored in mp_IHpokeTimeSmoothed
    spImHandler2D(float) mp_IHpokeStdDevSmoothed;

    // == Helper functions

    // Does processing on the recorded or set response.
    // - Saves the raw response
    // - Measures the PtV and the RMS of the response
    // - Saves the fits filename and stats of the response to a text file
    void processRawResponse();

    // Records a sequence from the WFS.
    // IHdst is the destination image
    // numFrames determines the length of the recording
    // dstFrameOffset offsets the writing position in frames
    // timestampDst will be populated with the absolute acquisition
    //      time of each frame. The dstFrameOffset applies here
    //      as well.
    void recordSequence(
        spImHandler2D(float) IHdst,
        int numFrames,
        int dstFrameOffset = 0,
        double* timestampDst = nullptr);

    // Collects all samples from all sequences and packs them
    // into a time-sorted vector
    std::vector<std::pair<double, float>> getTimeSortedResponseSamples();
};

#endif // MLS_SEQUENCEMANAGER_HPP