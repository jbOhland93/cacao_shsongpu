#include "../../AOloopControl_IOtools/shs_on_gpu/util/ImageHandler2D.hpp"
#include <errno.h>
#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <fstream>

enum PokePattern : int32_t {
    HOMOGENEOUS = 0,
    SINE = 1,
    CHECKERBOARD = 2
};

// A class for recording the hardware latency of slow DMs
// Works by fitting the expected response to the WFS image
class MLS_Recorder
{
public:
    MLS_Recorder(
        IMAGE* dmstream,        // Stream of the DM input
        IMAGE* wfsstream,       // Stream of the WFS output
        float fpsMeasTime,      // Timeframe for wfs framerate estimation
        uint32_t pokePattern,   // Poke pattern:
        float maxActStroke,     // Maximum actuator stroke in pattern
        uint32_t numPokes,      // number of iterations
        uint32_t framesPerPoke, // number of frames per iteration
        FUNCTION_PARAMETER_STRUCT* fps, // process related fps
        bool saveRaw);          // If true, each iterations frames is saved to fits
    ~MLS_Recorder();

    // Launches the recording sequence
    void recordDo();

    // Result getters
    float getFPS_Hz() { return m_FPS_Hz; }
    float getHWdelay_frames() { return m_hwDelay_frames; }
    float getHWdelay_us() { return m_hwDelay_us; }
    float getRiseTime0to90_frames() { return m_riseTime10to90_frames; }
    float getRiseTime0to90_us() { return m_riseTime10to90_us; }
    float getHWlatency_frames() { return m_hwLatency_frames; }
    float getHWlatency_us() { return m_hwLatency_us; }

private:
    // Parameters
    float m_fpsMeasurementTime;
    PokePattern m_pokePattern;
    float m_maxStroke;
    uint32_t m_numPokes;
    uint32_t m_framesPerPoke;
    uint32_t m_framesPriorToPoke;
    FUNCTION_PARAMETER_STRUCT* mp_fps;
    bool m_saveRaw;

    // Output files
    std::ofstream m_cacaovarsOutput;
    std::ofstream m_pokeAmpOutput;
    std::ofstream m_smoothedAmpOutput;

    // Intermediate results
    double m_wfsdt_us;
    double m_FPS_Hz;
    int m_iteration = 0;

    // Results
    float m_hwDelay_frames;         // Delay from poke to first movement
    float m_hwDelay_us;             // Delay from poke to first movement
    float m_riseTime10to90_us;      // Rise time from 10% to 90% of stroke
    float m_riseTime10to90_frames;  // Rise time from 10% to 90% of stroke
    float m_hwLatency_frames;       // Delay from poke to 90% stroke
    float m_hwLatency_us;           // Delay from poke to 90% of stroke

    // == IMAGE HANDLERS ==
    // Adopted DM input stream
    spImHandler2D(float) mp_IHdm;
    spImHandler2D(float) mp_IHdmPoke0;
    spImHandler2D(float) mp_IHdmPoke1;
    // Adopted WFS output stream
    spImHandler2D(float) mp_IHwfs;
    // Sequence for measuring the poke respnse
    spImHandler2D(float) mp_IHpokeResponseSequence;
    // Single WF, containing the expected poke response
    spImHandler2D(float) mp_IHpokeResponse;
    // Sequence containing frames of one poke
    spImHandler2D(float) mp_IHLatencySequence;
    // 3D image, containing the decomposed poke evolutions
    // The corresponding timestamps are stored in mp_IHpokeTimesRel
    spImHandler2D(float) mp_IHpokeAmps;
    // Contains the relative timestamps of the pokes in ns
    // The corresponding amplitudes are stored in mp_IHpokeAmps
    // Format is double as the time is measured in ns since the last epoch
    // Here, float would not yield sufficient precision.
    spImHandler2D(double) mp_IHpokeTimesRel;
    // Smoothed avg poke response amplitude
    // The corresponding relative times are stored in mp_IHpokeTimeSmoothed
    spImHandler2D(float) mp_IHpokeAmpSmoothed;
    // Relative times in ns of the smoothed avg poke response amplitude
    // The corresponding amplitudes are stored in mp_IHpokeAmpSmoothed
    spImHandler2D(double) mp_IHpokeTimeSmoothed;

    // == STATE MACHINE ==
    // Internal status
    enum RECSTATE {
        ERROR,
        CONSTRUCTION,
        INITIALIZING,
        READY,
        MEASURE_FRAMERATE,
        RECORD_POKE_RESPONSE,
        EVAL_POKE_RESPONSE,
        RECORD_LATENCY_SEQUENCE,
        DECOMPOSE_LATENCY_SEQUENCE,
        EVAL_LATENCY,
        FINISH};
    std::string recstateToStr(RECSTATE state);
    // Current internal state
    RECSTATE m_state = RECSTATE::CONSTRUCTION;
    void throwError(std::string msg);
    void switchState(RECSTATE newState);
    // State functions
    void execStateError();
    void execStateInitializing();
    void execStateReady();
    void execStateMeasureFPS();
    void execStateRecordPokeResponse();
    void execStateEvalPokeResponse();
    void execStateRecordLatencySequence();
    void execStateDecomposeLatencySequence();
    void execStateEvalLatency();
    void execStateFinish();
    
    // == HELPER FUNCTIONS ==
    // Copies the channel values corresponding to the
    // selected poke to the DM without triggering an update.
    void preloadDM(bool poke);
    // Updates the dm without copying new data.
    void triggerDM();
    // Copies the channel values corresponding to the
    // selected poke to the DM and triggers an update.
    // For more immediate updates, use preloadDM() first
    // and triggerDM() at the desired poit in time
    void pokeDM(bool poke) { preloadDM(poke); triggerDM(); }
    // Waits for a given number of WFS frames
    void waitWFSframes(int frames);
    // Record a sequence of frames into a new image
    template<typename T>
    void recordSequence(
        spImHandler2D(T) IHsrc,
        spImHandler2D(T) IHdst,
        int numFrames,
        int dstFrameOffset = 0,
        double* timestampDst = nullptr);
};

template<typename T>
void MLS_Recorder::recordSequence(
        spImHandler2D(T) IHsrc,
        spImHandler2D(T) IHdst,
        int numFrames,
        int dstFrameOffset,
        double* timestampDst)
{
    bool recTime = timestampDst != nullptr;
    // Get raw data pointers
    T* src = IHsrc->getWriteBuffer();
    T* dst = IHdst->getWriteBuffer();
    double* dstT = timestampDst;
    // Apply frame offset
    dst += IHsrc->mNumPx * dstFrameOffset;
    if (recTime)
        dstT += dstFrameOffset;
    // Record the sequence
    for (int i = 0; i < numFrames; i++)
    {
        IHsrc->waitForNextFrame();
        // Record time
        if (recTime)
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto nanoseconds_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
            dstT[i] = (double) nanoseconds_since_epoch.count();
        }
        // Copy pixels
        for (int k = 0; k < IHsrc->mNumPx; k++)
            dst[k] = src[k];
        // Proceed to next frame
        dst += mp_IHwfs->mNumPx;
    }
}
