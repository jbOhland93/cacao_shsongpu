#ifndef MLS_RECORDER_HPP
#define MLS_RECORDER_HPP

#include "MLS_PokePattern.hpp"
#include "MLS_ResultManager.hpp"
#include "MLS_SequenceManager.hpp"
#include "MLS_DMmanager.hpp"
#include <errno.h>
#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <fstream>

// A state machine for recording the hardware latency of slow DMs
// Works by fitting the expected response to the WFS image
class MLS_Recorder
{
public:
    MLS_Recorder(
        FUNCTION_PARAMETER_STRUCT* fps, // process related fps
        IMAGE* dmstream,        // Stream of the DM input
        IMAGE* wfsstream,       // Stream of the WFS output
        float fpsMeasTime,      // Timeframe for wfs framerate estimation
        uint32_t pokePattern,   // Poke pattern:
        float maxActStroke,     // Maximum actuator stroke in pattern
        uint32_t numPokes,      // number of iterations
        uint32_t framesPerPoke, // number of frames per iteration
        bool saveRaw);          // If true, each iterations frames is saved to fits

    // Launches the recording sequence
    void recordDo();

private:
    // Parameters
    IMAGE* mp_dmImage;
    IMAGE* mp_wfsImage;
    FUNCTION_PARAMETER_STRUCT* mp_fps;
    float m_fpsMeasurementTime;
    PokePattern m_pokePattern;
    float m_maxStroke;
    uint32_t m_numPokes;
    uint32_t m_framesPerPoke;
    bool m_saveRaw;

    int m_iteration = 0;

    // DM manager
    std::shared_ptr<MLS_DMmanager> mp_dmMngr;
    // Sequence manager
    std::shared_ptr<MLS_SequenceManager> mp_seqMngr;
    // Result manager
    std::shared_ptr<MLS_ResultManager> mp_resultMngr;

    // == STATE MACHINE ==
    // Internal status
    enum RECSTATE {
        ERROR,
        CONSTRUCTION,
        INITIALIZING,
        READY,
        MEASURE_FRAMERATE,
        RECORD_POKE_RESPONSE,
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
    void execStateRecordLatencySequence();
    void execStateDecomposeLatencySequence();
    void execStateEvalLatency();
    void execStateFinish();
    
    // == HELPER FUNCTIONS ==
    // Pokes the DM and waits one sequence duration until it settled
    void pokeAndSettle(bool poke);
};

#endif // MLS_RECORDER_HPP
