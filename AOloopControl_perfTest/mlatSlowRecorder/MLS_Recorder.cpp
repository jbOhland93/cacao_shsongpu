#include "MLS_Recorder.hpp"
#include <map>
#include <iostream>
#include <thread>

using namespace std::chrono;

#define LOGSTREAM std::cout << "MLS_Recorder: "
#define ERRSTREAM std::cerr << "MLS_Recorder error: "

MLS_Recorder::MLS_Recorder(
        IMAGE* dmstream,        // Stream of the DM input
        IMAGE* wfsstream,       // Stream of the WFS output
        float fpsMeasTime,      // Timeframe for wfs framerate estimation
        uint32_t pokePattern,   // Poke pattern:
        float maxActStroke,     // Maximum actuator stroke in pattern
        uint32_t numPokes,      // number of iterations
        uint32_t framesPerPoke, // number of frames per iteration
        FUNCTION_PARAMETER_STRUCT* fps, // process related fps
        bool saveRaw)           // If true, each iterations frames is saved to fits
        :   m_maxStroke(maxActStroke),
            m_numPokes(numPokes),
            m_framesPerPoke(framesPerPoke),
            m_framesPriorToPoke((uint32_t) (framesPerPoke * 0.1)),
            mp_fps(fps),
            m_saveRaw(saveRaw)
{
    // verify poke pattern option
    switch ((PokePattern) pokePattern)
    {
    case PokePattern::HOMOGENEOUS: break;
    case PokePattern::SINE: break;
    case PokePattern::CHECKERBOARD: break;
    default:
        throw std::runtime_error("MLS_Recorder::MLS_Recorder: poke pattern not recognized.");
    }
    m_pokePattern = (PokePattern) pokePattern;

    // Verify fps measurement time
    if (fpsMeasTime < 1)
        throw std::runtime_error("MLS_Recorder::MLS_Recorder: framerate measurement time should be >1.");
    m_fpsMeasurementTime = fpsMeasTime;

    // Adopt image streams
    mp_IHdm = ImageHandler2D<float>::newHandler2DAdoptImage(dmstream->name);
    mp_IHwfs = ImageHandler2D<float>::newHandler2DAdoptImage(wfsstream->name);

    // Switch to first state
    switchState(RECSTATE::INITIALIZING);
}

MLS_Recorder::~MLS_Recorder()
{
    m_cacaovarsOutput.close();
    m_pokeAmpOutput.close();
    m_smoothedAmpOutput.close();
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
        case RECSTATE::EVAL_POKE_RESPONSE: return "EVAL_POKE_RESPONSE";
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
              << recstateToStr(m_state) << "':" << std::endl
              << "\t" << msg << std::endl;
    switchState(RECSTATE::ERROR);
}



void MLS_Recorder::switchState(RECSTATE newState)
{
    LOGSTREAM << "Switching from state '" << recstateToStr(m_state)
              << "' to state '" << recstateToStr(newState) << "'."
              << std::endl;
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
        case RECSTATE::EVAL_POKE_RESPONSE: return execStateEvalPokeResponse();
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
    // Create output files
    std::string cacaovarsName = mp_fps->md->datadir;
    cacaovarsName += "/cacaovars.bash";
    m_cacaovarsOutput = std::ofstream(cacaovarsName);
    if (!m_cacaovarsOutput.is_open())
        throwError(cacaovarsName + "could not be opened.");

    std::string pokeAmpOutputName = mp_fps->md->datadir;
    pokeAmpOutputName += "/pokeAmpOutput.dat";
    m_pokeAmpOutput = std::ofstream(pokeAmpOutputName);
    if (!m_pokeAmpOutput.is_open())
        throwError(pokeAmpOutputName + "could not be opened.");
    m_pokeAmpOutput << "# Responses of each individual iteration" << std::endl;
    m_pokeAmpOutput << "# " << std::endl;
    m_pokeAmpOutput << "# 1: frame index, 0=first frame after poke" << std::endl;
    m_pokeAmpOutput << "# 2: time in us relative to poke time" << std::endl;
    m_pokeAmpOutput << "# 3: normalized response" << std::endl;
    m_pokeAmpOutput << "# Note: Each sequence is followed by a line containing information" << std::endl
                    << " about the frame that marks the last one not being conained in the" << std::endl
                    << " [.9|1.1] inverval:" << std::endl;
    m_pokeAmpOutput << "# \t #\tFrameIndexOf0.9Response\tTimeOf0.9ResponseInSeconds" << std::endl;

    std::string smoothedAmpOutputName = mp_fps->md->datadir;
    smoothedAmpOutputName += "/smoothedPokeAmps.dat";
    m_smoothedAmpOutput = std::ofstream(smoothedAmpOutputName);
    if (!m_smoothedAmpOutput.is_open())
        throwError(smoothedAmpOutputName + "could not be opened.");
    m_smoothedAmpOutput << "# Temporal averaging of all respones" << std::endl;

    // Make DM patterns
    std::string dmStreamName = mp_IHdm->getImage()->name;

    std::string poke0name = dmStreamName;
    poke0name += "_mlatPoke0";
    mp_IHdmPoke0 = ImageHandler2D<float>::newHandler2DfrmImage(
        poke0name,
        mp_IHdm->getImage());
    float* dptr0 = mp_IHdmPoke0->getWriteBuffer();

    std::string poke1name = dmStreamName;
    poke1name += "_mlatPoke1";
    mp_IHdmPoke1 = ImageHandler2D<float>::newHandler2DfrmImage(
        poke1name,
        mp_IHdm->getImage());
    float* dptr1 = mp_IHdmPoke1->getWriteBuffer();

    for (int iy = 0; iy < mp_IHdm->mHeight; iy++)
        for (int ix = 0; ix < mp_IHdm->mWidth; ix++)
        {
            int i = mp_IHdm->mWidth * iy + ix;
            dptr0[i] = 0;
            switch (m_pokePattern)
            {
            case PokePattern::HOMOGENEOUS:
                dptr1[i] = m_maxStroke;
                break;
            case PokePattern::CHECKERBOARD:
                dptr1[i] = m_maxStroke * (( ix + iy % 2 ) % 2);
                break;
            case PokePattern::SINE:
                dptr1[i] = m_maxStroke
                            * cos(20 * ix/mp_IHdm->mWidth)
                            * cos(20 * iy/mp_IHdm->mWidth);
                break;
            default:
                throwError("Unknown PokePattern.");
            }
        }
    mp_IHdmPoke0->updateWrittenImage();
    mp_IHdmPoke1->updateWrittenImage();
    m_pokeAmpOutput << "# Max actuator stroke = " << m_maxStroke << std::endl;
    m_smoothedAmpOutput << "# Max actuator stroke = " << m_maxStroke << std::endl;
    m_pokeAmpOutput << "# Poke pattern = " << m_pokePattern << std::endl;
    m_smoothedAmpOutput << "# Poke pattern = " << m_pokePattern << std::endl;

    // Initialize poke response sequence
    std::string pokeSeqName = dmStreamName;
    pokeSeqName += "_pokeRespSeq";
    mp_IHpokeResponseSequence = ImageHandler2D<float>::newImageHandler2D(
        pokeSeqName,
        mp_IHwfs->mWidth,
        mp_IHwfs->mHeight,
        m_framesPerPoke * 2); // One set of frames prior to poke, one set after
    
    // Initialize poke response
    std::string pokeRespName = dmStreamName;
    pokeRespName += "_pokeResp";
    mp_IHpokeResponse = ImageHandler2D<float>::newHandler2DfrmImage(
        pokeRespName, mp_IHwfs->getImage());

    // Initialize latency sequence
    std::string latSeqName = dmStreamName;
    latSeqName += "_lateSeq";
    mp_IHLatencySequence = ImageHandler2D<float>::newImageHandler2D(
        latSeqName, mp_IHwfs->mWidth, m_framesPerPoke);

    // Initialize poke amplitude collection image
    std::string pokeAmpsName = dmStreamName;
    pokeAmpsName += "_pokeAmps";
    mp_IHpokeAmps = ImageHandler2D<float>::newImageHandler2D(
        pokeAmpsName, m_framesPerPoke, m_numPokes);
    std::string pokeTimesName = dmStreamName;
    pokeTimesName += "_pokeTimesRel";
    mp_IHpokeTimesRel = ImageHandler2D<double>::newImageHandler2D(
        pokeTimesName, m_framesPerPoke, m_numPokes);

    // Create Sequence cubes and result images
    switchState(RECSTATE::READY);
}



void MLS_Recorder::execStateReady()
{
    LOGSTREAM << "Initialiaztion done. Ready to start evaluation."
              << std::endl;
}



void MLS_Recorder::execStateMeasureFPS()
{
    LOGSTREAM << "Measuring frame rate over "
              << m_fpsMeasurementTime << " sec." << std::endl;

    // Sync with WFS START
    mp_IHwfs->waitForNextFrame();
    auto t_start = high_resolution_clock::now();
    long wfscntstart = mp_IHwfs->getCnt0();

    // Wait for given duration
    std::this_thread::sleep_for(
        milliseconds((int) m_fpsMeasurementTime * 1000));

    // Sync with WFS END
    mp_IHwfs->waitForNextFrame();
    auto t_end = high_resolution_clock::now();
    long wfscntend = mp_IHwfs->getCnt0();


    // Evaluate
    if ( wfscntend - wfscntstart < 5)
        throwError("Number of frames in time window too small -> cannot proceed.");
    else
    {
        auto timeDelta = t_end - t_start;
        double timeDelta_ns = duration_cast<nanoseconds>(timeDelta).count();
        long numFrames = wfscntend - wfscntstart;

        m_wfsdt_us = timeDelta_ns / numFrames / 1e3;
        m_FPS_Hz = 1e9 * numFrames / timeDelta_ns;
        
        std::cout << "\t wfs dt = " << m_wfsdt_us << "us "
                  << "(" << m_FPS_Hz << "Hz)" << std::endl;

        m_cacaovarsOutput << "export CACAO_WFSFRATE=" << m_FPS_Hz << std::endl;
        m_pokeAmpOutput << "# WFS dt: " << m_wfsdt_us << "us (FPS=" << m_FPS_Hz << "Hz)" << std::endl;
        m_smoothedAmpOutput << "# WFS dt: " << m_wfsdt_us << "us (FPS=" << m_FPS_Hz << "Hz)" << std::endl;
        // Done! Switch to next state.
        switchState(RECSTATE::RECORD_POKE_RESPONSE);
    }
}



void MLS_Recorder::execStateRecordPokeResponse()
{   
    // Reset DM and wait until it settled
    pokeDM(false);
    waitWFSframes(m_framesPerPoke);
    // Record sequence prior to poke
    recordSequence<float>(
        mp_IHwfs, mp_IHpokeResponseSequence, m_framesPerPoke, 0);

    // Poke DM and wait until it settled
    pokeDM(true);
    waitWFSframes(m_framesPerPoke);
    // Record sequence after the poke
    recordSequence<float>(
        mp_IHwfs, mp_IHpokeResponseSequence, m_framesPerPoke, m_framesPerPoke);

    // On to the next tate
    switchState(RECSTATE::EVAL_POKE_RESPONSE);
}



void MLS_Recorder::execStateEvalPokeResponse()
{
    float* src = mp_IHpokeResponseSequence->getWriteBuffer();
    float* dst = mp_IHpokeResponse->getWriteBuffer();

    int sequenceFrames = mp_IHpokeResponseSequence->mDepth;
    int pxPerFrame = mp_IHpokeResponse->mNumPx;
    
    // Accumulate differences
    for (int frmIdx = 0; frmIdx < sequenceFrames; frmIdx++)
        for (int pxIdx = 0; pxIdx < pxPerFrame; pxIdx++)
        {   
            // First half of sequence: pre poke
            // 2nd half of sequence: post poke
            if (frmIdx < sequenceFrames/2)
                dst[pxIdx] -= src[frmIdx*pxPerFrame + pxIdx];
            else
                dst[pxIdx] += src[frmIdx*pxPerFrame + pxIdx];
        }
    
    // Subtract mean
    float mean = 0;
    for (int pxIdx = 0; pxIdx < pxPerFrame; pxIdx++)
        mean += dst[pxIdx];
    mean /= pxPerFrame;

    // Normalize to RMS
    float rms = 0;
    for (int pxIdx = 0; pxIdx < pxPerFrame; pxIdx++)
    {
        dst[pxIdx] -= mean;
        rms += dst[pxIdx]*dst[pxIdx];
    }
    rms = sqrt(rms/pxPerFrame);
    for (int pxIdx = 0; pxIdx < pxPerFrame; pxIdx++)
        dst[pxIdx] /= rms;        

    // Done - update Image
    mp_IHpokeResponse->updateWrittenImage();

    // On to the next tate
    switchState(RECSTATE::RECORD_LATENCY_SEQUENCE);
}



void MLS_Recorder::execStateRecordLatencySequence()
{
    LOGSTREAM   << "Recording latency sequence "
                << m_iteration << " of " << m_numPokes
                << std::endl;

    // == Preparations
    // Determine the interframe delay between the wfs frame and the poke
    double interframePokeDelay_frm = m_iteration / (double) m_numPokes;
    double interframePokeDelay_ns = m_wfsdt_us * 1000 * interframePokeDelay_frm;

    // Get the pointer location for the timestamps
    double* timestampDst = mp_IHpokeTimesRel->getWriteBuffer();
    // Skip to line of the current iteration
    timestampDst += mp_IHpokeTimesRel->mWidth * m_iteration;


    // == Reset DM and wait until it settled
    pokeDM(false);
    waitWFSframes(m_framesPerPoke);
    // Preload DM
    preloadDM(true);
    // == Record the pre-poke sequence
    recordSequence(mp_IHwfs, mp_IHLatencySequence,
        m_framesPriorToPoke, 0,
        timestampDst);

    // == Wait for the specific time to perform the poke
    // Get the time in ns since epoch from the last frame prior to the poke
    double prePokeFrameTimestamp_ns = timestampDst[m_framesPriorToPoke-1];
    // Get the current time in nanoseconds since the epoch.
    auto current_time = high_resolution_clock::now();
    double current_time_ns =
        (double) duration_cast<nanoseconds>(
            current_time.time_since_epoch()).count();
    // Calculate the desired poke time.
    auto pokeTime = current_time + nanoseconds((long)
        (prePokeFrameTimestamp_ns - current_time_ns + interframePokeDelay_ns));
    // Wait until this time is reached
    while (high_resolution_clock::now() < pokeTime);
    
    // == Apply the poke!
    triggerDM();
    // == Record the post-poke sequence
    recordSequence(mp_IHwfs, mp_IHLatencySequence,
        m_framesPerPoke - m_framesPriorToPoke, m_framesPriorToPoke,
        timestampDst);
    mp_IHLatencySequence->updateWrittenImage();

    if (m_saveRaw)
    {
        
        char* iterString = new char[10];
        sprintf(iterString, "%04d", m_iteration);
        std::string fname = "mlat-testC-";
        fname += iterString;

        mp_IHLatencySequence->saveToFPSdataDir(mp_fps, fname);
        delete[] iterString;
    }

    // == Subtract the poke time from the timestamps
    for (int i = 0; i < m_framesPerPoke; i++)
        timestampDst[i] -= prePokeFrameTimestamp_ns + interframePokeDelay_ns;

    // On to the next state
    switchState(RECSTATE::DECOMPOSE_LATENCY_SEQUENCE);
}



void MLS_Recorder::execStateDecomposeLatencySequence()
{
    LOGSTREAM   << "Decomposing latency sequence "
                << m_iteration << " of " << m_numPokes
                << std::endl;

    // == Preparations
    // Get the source of the pokes to decompose
    float* wfSrc = mp_IHLatencySequence->getWriteBuffer();
    // Get response ptr
    float* resp = mp_IHpokeResponse->getWriteBuffer();
    // Get the destination for the decomposition
    float* ampDst = mp_IHpokeAmps->getWriteBuffer();
    // Skip to line of the current iteration
    ampDst += mp_IHpokeAmps->mWidth * m_iteration;

    // == Calculate baseline
    int pxPerFrame = mp_IHLatencySequence->mWidth;
    // Init baseline array
    float* baseline = new float[pxPerFrame];
    for (int px = 0; px < pxPerFrame; px++)
        baseline[px] = 0;
    // Add up frames prior to poke
    for (int frame = 0; frame < m_framesPriorToPoke; frame++)
        for (int px = 0; px < pxPerFrame; px++)
            baseline[px] += wfSrc[frame * pxPerFrame + px];
    // Divide by number of frames
    for (int px = 0; px < pxPerFrame; px++)
        baseline[px] /= m_framesPriorToPoke;
    
    // == Decompose sequence
    float amplitude;
    float curWFSval;
    float avgSettleValue = 0;
    int settleValueFrames = 0;
    for (int frame = 0; frame < m_framesPerPoke; frame++)
    {
        amplitude = 0;
        for (int px = 0; px < pxPerFrame; px++)
        {
            curWFSval = wfSrc[frame * pxPerFrame + px];
            amplitude +=  (curWFSval - baseline[px]) * resp[px];
        }
        ampDst[frame] = amplitude;
        if (frame >= 0.9*m_framesPerPoke)
        {
            avgSettleValue += amplitude;
            settleValueFrames++;
        }
    }
    avgSettleValue /= settleValueFrames;
    // == Normalize to last values
    for (int frame = 0; frame < m_framesPerPoke; frame++)
        ampDst[frame] /= avgSettleValue;

    // == Write result to file
    double* timestamps = mp_IHpokeTimesRel->getWriteBuffer();
    timestamps += mp_IHpokeTimesRel->mWidth * m_iteration;
    long i_90 = 0;
    float t_90 = 0;
    for (int frame = 0; frame < m_framesPerPoke; frame++)
    {
        m_pokeAmpOutput << frame - (int)m_framesPriorToPoke << "\t";
        m_pokeAmpOutput << timestamps[frame] << "\t";
        m_pokeAmpOutput << ampDst[frame] << std::endl;
        if (ampDst[frame] < 0.9)
        {
            i_90 = frame;
            t_90 = timestamps[frame];
        }
    }
    m_pokeAmpOutput << "#\t" << i_90 << "\t" << t_90/1e6 << std::endl;
    
    // == Cleanup
    delete[] baseline;

    // On to the next state
    m_iteration++;
    if (m_iteration < m_numPokes)
        switchState(RECSTATE::RECORD_LATENCY_SEQUENCE);
    else
        switchState(RECSTATE::EVAL_LATENCY);
}



void MLS_Recorder::execStateEvalLatency()
{
    LOGSTREAM   << "Starting latency evaluation." << std::endl;

    // Get amplitude array
    float* ampArr = mp_IHpokeAmps->getWriteBuffer();
    // Get the timestamp array
    double* timestampArr = mp_IHpokeTimesRel->getWriteBuffer();

    // == Sort the data points by timestamp
    // Pairing timestamps and amplitudes
    std::vector<std::pair<double, float>> samples;
    for (int i = 0; i < mp_IHpokeAmps->mNumPx; i++)
        samples.push_back({timestampArr[i], ampArr[i]});
    // Sorting based on timestamps
    std::sort(samples.begin(), samples.end());

    // == Span time grid over recorded window
    double timeResolution_ns = m_wfsdt_us * 1000 / 10;  // 10 samples per frame
    double minTime_ns = samples[0].first;
    double maxTime_ns = samples[samples.size() - 1].first;
    int numSamples = ceil((maxTime_ns - minTime_ns) / timeResolution_ns);

    // Initialize smoothed poke amplitude image
    std::string smoothedTimeName = mp_IHdm->getImage()->name;
    smoothedTimeName += "_smoothedPokeTimes";
    mp_IHpokeTimeSmoothed = ImageHandler2D<double>::newImageHandler2D(
        smoothedTimeName, numSamples, 1);
    // Initialize smoothed time amplitude image
    std::string smoothedAmpName = mp_IHdm->getImage()->name;
    smoothedAmpName += "_smoothedPokeAmps";
    mp_IHpokeAmpSmoothed = ImageHandler2D<float>::newImageHandler2D(
        smoothedAmpName, numSamples, 1);
    
    double* dstTime = mp_IHpokeTimeSmoothed->getWriteBuffer();
    float* dstAmp = mp_IHpokeAmpSmoothed->getWriteBuffer();
    
    // == Convolve sequence with gaussian kernel
    double kernelStdDev_ns = m_wfsdt_us * 1000 / 4;
    double kernelHalfSize_ns = kernelStdDev_ns * 3;
    double kernelDenominator = -2*kernelStdDev_ns*kernelStdDev_ns;

    LOGSTREAM   << "Applying floating avg on recorded sequences." << std::endl;
    std::cout   << "\t Time grid sample spacing: "
                << timeResolution_ns/1000 << "us (1/10 frame spacing) " << std::endl;
    std::cout   << "\t Using gaussian kernel with a stddev of "
                << kernelStdDev_ns / 1000 << "us (1/4 frame spacing)" << std::endl;

    // Write output file header
    m_smoothedAmpOutput << "# " << std::endl;
    m_smoothedAmpOutput << "# Temporal resolution: "
                        << timeResolution_ns/1000
                        << "us (1/10 frame spacing)" << std::endl;
    m_smoothedAmpOutput << "# Temporal smoothing stdDev (gaussian): "
                        << kernelStdDev_ns / 1000
                        << "us (1/4 frame spacing)" << std::endl;
    m_smoothedAmpOutput << "# " << std::endl;
    m_smoothedAmpOutput << "# 1: Time relative to poke command in ns" << std::endl;
    m_smoothedAmpOutput << "# 2: Normalized response" << std::endl;
    m_smoothedAmpOutput << "# " << std::endl;
    
    for (int i = 0; i < numSamples; i++)
    {
        double kernelCenterTime_ns = minTime_ns + i * timeResolution_ns;
        double windowStart_ns = kernelCenterTime_ns - kernelHalfSize_ns;
        double windowEnd_ns = kernelCenterTime_ns + kernelHalfSize_ns;

        double kernelSum = 0;
        double integral = 0;
        for (int k = 0; k < samples.size(); k++)
        {
            std::pair<double, float> sample = samples.at(k);
            double delT = sample.first - kernelCenterTime_ns;
            double kernelValue = exp(delT*delT/kernelDenominator);
            kernelSum += kernelValue;
            integral += sample.second * kernelValue;
        }
        // Write output to images
        dstTime[i] = kernelCenterTime_ns;
        dstAmp[i] = (float) integral / kernelSum;
        // Write output to file
        m_smoothedAmpOutput << dstTime[i] << "\t" << dstAmp[i] << std::endl;
    }
    mp_IHpokeTimeSmoothed->updateWrittenImage();
    mp_IHpokeAmpSmoothed->updateWrittenImage();

    // Find keypoints in response curve
    std::cout   << "\t Identifying keypoints in curve ..." << std::endl;
    double t_10 = 0;
    double t_70 = 0;
    double t_90_110 = 0;
    for (int i = 0; i < numSamples; i++)
    {
        double time = dstTime[i];
        double amp = dstAmp[i];
        if (amp < 0.1)
            t_10 = time;
        if (amp < 0.7)
            t_70 = time;
        if (amp < 0.9 || amp > 1.1)
            t_90_110 = time;
    }
    // Assuming a linear rise time calculate beginning of movement
    m_hwDelay_us = (t_10 - (t_70 - t_10) / 6) / 1000;
    // Calc 10-90% rise time, including setteling between 90%-110%
    m_riseTime10to90_us = (t_90_110 - t_10) / 1000;
    // Calc hw latency
    m_hwLatency_us = t_90_110 / 1000;

    // Convert to frames
    m_hwDelay_frames = m_hwDelay_us / m_wfsdt_us;
    m_riseTime10to90_frames = m_riseTime10to90_us / m_wfsdt_us;
    m_hwLatency_frames = m_hwLatency_us / m_wfsdt_us;

    m_cacaovarsOutput << "export CACAO_LATENCYHARDWFR=" << m_hwLatency_frames << std::endl;
    m_cacaovarsOutput << "export CACAO_LATENCYHARDWUS=" << m_hwLatency_us << std::endl;
    m_cacaovarsOutput << "export CACAO_DELAYHARDWFR=" << m_hwDelay_frames << std::endl;
    m_cacaovarsOutput << "export CACAO_DELAYHARDWUS=" << m_hwDelay_us << std::endl;
    m_cacaovarsOutput << "export CACAO_DMRISETIMEFR=" << m_riseTime10to90_frames << std::endl;
    m_cacaovarsOutput << "export CACAO_DMRISETIMEUS=" << m_riseTime10to90_us << std::endl;

    switchState(RECSTATE::FINISH);
}



void MLS_Recorder::execStateFinish()
{
    LOGSTREAM << "Evaluation done!" << std::endl;
    LOGSTREAM << "############################" << std::endl;
    LOGSTREAM << "ToDo: Implement fits saving." << std::endl;
    LOGSTREAM << "############################" << std::endl;
}


// ===========================================================
// Helper functions
// ===========================================================

void MLS_Recorder::preloadDM(bool poke)
{
    if (poke)
        mp_IHdm->cpy(mp_IHdmPoke1->getImage(), false);
    else
        mp_IHdm->cpy(mp_IHdmPoke0->getImage(), false);
}



void MLS_Recorder::triggerDM()
{
    mp_IHdm->updateWrittenImage();
}



void MLS_Recorder::waitWFSframes(int frames)
{
    for (int i = 0; i < frames; i++)
        mp_IHwfs->waitForNextFrame();
}