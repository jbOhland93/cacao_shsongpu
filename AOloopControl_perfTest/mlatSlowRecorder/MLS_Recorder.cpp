#include "MLS_Recorder.hpp"
#include <map>
#include <iostream>
#include <thread>

using namespace std::chrono;

#define LOGSTREAM std::cout << "MLS_Recorder: "
#define ERRSTREAM std::cerr << "MLS_Recorder error: "

MLS_Recorder::MLS_Recorder(
        FUNCTION_PARAMETER_STRUCT* fps,     // process relatef fps
        IMAGE* dmstream,                    // Stream of the DM input
        IMAGE* wfsstream,                   // Stream of the WFS output
        int64_t skipMFramerate,             // If true, the FPS measurement prior to the latency is skipped
        float fpsMeasTime,                  // Timeframe for wfs framerate estimation
        uint32_t numPokes,                  // number of iterations
        uint32_t framesPerPoke,             // number of frames per iteration
        bool saveRaw,                       // If true, each iterations frames is saved to fits
        int32_t pokePatternType,            // Poke pattern type
        const char* customPatternStream,    // Name of pattern stream in shm
        uint32_t customPatternSliceIdx,     // Index of the shm pattern slice to be poked
        float patternToStrokeMul,           // Pattern-to-poke factor
        bool useCustomResponseStream,       // Don't record the response but use custom one
        const char*customResponseStream,    // Name of response stream in shm
        uint32_t customResponseSliceIdx)    // Index of the shm response slice to be poked
        :   mp_dmImage(dmstream),
            mp_wfsImage(wfsstream),
            mp_fps(fps),
            m_numPokes(numPokes),
            m_framesPerPoke(framesPerPoke),
            m_measureFramerate(!skipMFramerate),
            m_customPatternImageName(customPatternStream),
            m_customPokePatternIndex(customPatternSliceIdx),
            m_strokeMul(patternToStrokeMul),
            m_useCustomResponse(useCustomResponseStream),
            m_customResponseImageName(customResponseStream),
            m_customResponseIndex(customResponseSliceIdx),
            m_saveRaw(saveRaw)
{
    LOGSTREAM << "Constrution started...\n";

    // verify poke pattern option
    switch ((PokePattern) pokePatternType)
    {
    case PokePattern::SHMIM: break;
    case PokePattern::HOMOGENEOUS: break;
    case PokePattern::SINE: break;
    case PokePattern::CHECKERBOARD: break;
    case PokePattern::SQUARE: break;
    case PokePattern::HALFSQUARE: break;
    case PokePattern::DOUBLESQUARE: break;
    case PokePattern::XRAMP: break;
    case PokePattern::XHALF: break;
    case PokePattern::YRAMP: break;
    case PokePattern::YHALF: break;
    default:
        throw std::runtime_error("MLS_Recorder::MLS_Recorder: poke pattern type not recognized.");
    }
    m_pokePattern = (PokePattern) pokePatternType;

    // Verify fps measurement time
    if (fpsMeasTime < 1)
        throw std::runtime_error("MLS_Recorder::MLS_Recorder: framerate measurement time should be >1.");
    m_fpsMeasurementTime = fpsMeasTime;

    // Switch to first state
    switchState(RECSTATE::INITIALIZING);
}



void MLS_Recorder::recordDo()
{
    if (m_state != RECSTATE::READY)
        throwError("Cannot start recording - state != READY.");

    switchState(RECSTATE::MEASURE_FRAMERATE);
}



// ===========================================================
// Statemachine backbone
// ===========================================================

std::string MLS_Recorder:: recstateToStr(RECSTATE state)
{
    switch (state)
    {
        case RECSTATE::ERROR: return "ERROR";
        case RECSTATE::CONSTRUCTION: return "CONSTRUCTION";
        case RECSTATE::INITIALIZING: return "INITIALIZING";
        case RECSTATE::READY: return "READY";
        case RECSTATE::MEASURE_FRAMERATE: return "MEASURE_FRAMERATE";
        case RECSTATE::RECORD_POKE_RESPONSE: return "RECORD_POKE_RESPONSE";
        case RECSTATE::RECORD_LATENCY_SEQUENCE: return "RECORD_LATENCY_SEQUENCE";
        case RECSTATE::DECOMPOSE_LATENCY_SEQUENCE: return "DECOMPOSE_LATENCY_SEQUENCE";
        case RECSTATE::EVAL_LATENCY: return "EVAL_LATENCY";
        case RECSTATE::FINISH: return "FINISH";
        default: return "<unknown>";
    }
}



void MLS_Recorder::throwError(std::string msg)
{   
    ERRSTREAM << "an error occurred in state '"
              << recstateToStr(m_state) << "':\n"
              << "\t" << msg << "\n";
    switchState(RECSTATE::ERROR);
}



void MLS_Recorder::switchState(RECSTATE newState)
{
    LOGSTREAM << "Switching from state '" << recstateToStr(m_state)
              << "' to state '" << recstateToStr(newState) << "'.\n";
    RECSTATE oldState = m_state;
    m_state = newState;
    
    std::string unknownStateErrorMsg("unknown state encountered in switchState. ");
    unknownStateErrorMsg += "Old state: " + recstateToStr(oldState);

    switch (m_state)
    {
        case RECSTATE::ERROR: return execStateError();
        case RECSTATE::INITIALIZING: return execStateInitializing();
        case RECSTATE::READY: return execStateReady();
        case RECSTATE::MEASURE_FRAMERATE: return execStateMeasureFPS();
        case RECSTATE::RECORD_POKE_RESPONSE: return execStateRecordPokeResponse();
        case RECSTATE::RECORD_LATENCY_SEQUENCE: return execStateRecordLatencySequence();
        case RECSTATE::DECOMPOSE_LATENCY_SEQUENCE: return execStateDecomposeLatencySequence();
        case RECSTATE::EVAL_LATENCY: return execStateEvalLatency();
        case RECSTATE::FINISH: return execStateFinish();
        case RECSTATE::CONSTRUCTION:
            return throwError("Cannot switch to construction state.");
        default:
            return throwError(unknownStateErrorMsg);
    }
}



// ===========================================================
// State execution functions
// ===========================================================

void MLS_Recorder::execStateError()
{
    throw std::exception();
}



void MLS_Recorder::execStateInitializing()
{
    // Initialize result manager
    try {
        mp_resultMngr = std::make_shared<MLS_ResultManager>(
                                            mp_fps,
                                            m_pokePattern,
                                            m_strokeMul,
                                            m_numPokes,
                                            m_framesPerPoke);
        mp_dmMngr = std::make_shared<MLS_DMmanager>(
                                            mp_dmImage,
                                            m_pokePattern,
                                            m_customPatternImageName,
                                            m_customPokePatternIndex,
                                            m_strokeMul);
        mp_seqMngr = std::make_shared<MLS_SequenceManager>(
                                            mp_fps,
                                            mp_wfsImage,
                                            m_numPokes,
                                            m_framesPerPoke,
                                            m_saveRaw);
    }
    catch(const std::runtime_error& e)
    {
        throwError(e.what());
    }

    // Create Sequence cubes and result images
    switchState(RECSTATE::READY);
}



void MLS_Recorder::execStateReady()
{
    LOGSTREAM << "Initialiaztion done. Ready to start evaluation."
              << "\n";
}



void MLS_Recorder::execStateMeasureFPS()
{
    if (m_measureFramerate)
    {
        LOGSTREAM << "Measuring frame rate over "
                << m_fpsMeasurementTime << " sec.\n";

        try
        {
            double FPS = mp_seqMngr->measureFPS(m_fpsMeasurementTime);
            mp_resultMngr->setFPS(FPS);
            switchState(RECSTATE::RECORD_POKE_RESPONSE);
        }
        catch(const std::exception& e)
        {
            throwError(e.what());
        }
    }
    else
    {
        LOGSTREAM   << "Retrieved frame rate from fps: ";
        mp_resultMngr->setFrameratefromFPS();

        switchState(RECSTATE::RECORD_POKE_RESPONSE);
    }
}



void MLS_Recorder::execStateRecordPokeResponse()
{   
    if (!m_useCustomResponse)
    {   // Record the response of the given pattern
        // Un-Poke DM and record surface
        pokeAndSettle(false);
        mp_seqMngr->recordPokeResponse(false);
        
        // Un-Poke DM and record surface
        pokeAndSettle(true);
        mp_seqMngr->recordPokeResponse(true);

        // Evaluate poke response
        mp_seqMngr->evalPokeResponse();    
    }
    else // Do not record the response - use a predefined one instead
        mp_seqMngr->setPokeResponse(m_customResponseImageName, m_customResponseIndex);
    // On to the next tate
    switchState(RECSTATE::RECORD_LATENCY_SEQUENCE);
}



void MLS_Recorder::execStateRecordLatencySequence()
{
    LOGSTREAM   << "Recording latency sequence "
                << m_iteration << " of " << m_numPokes
                << "\n";
    // Always start in poked mode
    if (m_iteration == 0)
        pokeAndSettle(true);

    // == Preparations
    // Determine the interframe delay between the wfs frame and the poke
    double interframePokeDelay_frm = m_iteration / (double) m_numPokes;
    double interframePokeDelay_ns = mp_resultMngr->getWfsDt_us() * 1000 * interframePokeDelay_frm;
    std::cout << "\tInterframe poke delay = " << interframePokeDelay_ns/1000 << " us\n";
    // Alternate between flat and poked DM
    mp_dmMngr->preloadDM(m_iteration % 2);

    // == Record the poke sequence
    // Record pre-poke sequence
    auto pokeTime = mp_seqMngr->recordLatencyPokeSequence(
        m_iteration, interframePokeDelay_ns, false);
    // Wait for the specific time to perform the poke
    while (high_resolution_clock::now() < pokeTime);
    // Apply the poke!
    mp_dmMngr->triggerDM();
    // Record the post-poke sequence
    mp_seqMngr->recordLatencyPokeSequence(
        m_iteration, interframePokeDelay_ns, true);

    // On to the next state
    switchState(RECSTATE::DECOMPOSE_LATENCY_SEQUENCE);
}



void MLS_Recorder::execStateDecomposeLatencySequence()
{
    LOGSTREAM   << "Decomposing latency sequence "
                << m_iteration << " of " << m_numPokes
                << "\n";

    std::pair<double*, float*> decomposition;
    decomposition = mp_seqMngr->decomposeLastPokeSequence(m_iteration);
    mp_resultMngr->logRawAmplitude(decomposition.first, decomposition.second);

    // On to the next state
    m_iteration++;
    if (m_iteration < m_numPokes)
        switchState(RECSTATE::RECORD_LATENCY_SEQUENCE);
    else
    {   // Reset the DM, then switch state
        pokeAndSettle(false);
        switchState(RECSTATE::EVAL_LATENCY);
    }
}



void MLS_Recorder::execStateEvalLatency()
{
    LOGSTREAM   << "Starting latency evaluation.\n";

    std::cout   << "\t Identifying raw settling time "
                << "(worst case scenario) ...\n";
    double hwLatencyRaw_us = mp_seqMngr->calcRawSettlingLatency();
    mp_resultMngr->setHwLatencyRaw(hwLatencyRaw_us);

    std::cout   << "\t Apply temporal average to response sequences...\n";
    mp_seqMngr->smoothLatencyDecomposition(mp_resultMngr);

    std::cout   << "\t Extract times from smoothed response...\n";
    // Calc delay until first motion
    mp_resultMngr->setHWdelay(mp_seqMngr->calcHWdelay());
    // Calc 10-90% rise time, including setteling between 90%-110%
    mp_resultMngr->setRisetime(mp_seqMngr->calcRiseime());
    // Calc hw latency
    mp_resultMngr->setHwLatencySmoothed(mp_seqMngr->calcHWlatency());

    switchState(RECSTATE::FINISH);
}



void MLS_Recorder::execStateFinish()
{
    std::cout << "\n";
    LOGSTREAM << "Evaluation done! Latency measurement results:\n";

    mp_resultMngr->publishResults();
}


// ===========================================================
// Helper functions
// ===========================================================


void MLS_Recorder::pokeAndSettle(bool poke)
{
    mp_dmMngr->pokeDM(poke);
    mp_seqMngr->waitNumFrames(m_framesPerPoke);
}
