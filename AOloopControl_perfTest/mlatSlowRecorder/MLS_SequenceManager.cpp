#include "MLS_SequenceManager.hpp"
#include "MLS_ResultManager.hpp"
#include <thread>

using namespace std::chrono;

MLS_SequenceManager::MLS_SequenceManager(
        FUNCTION_PARAMETER_STRUCT* fps,
        IMAGE* wfsStream,
        uint32_t numPokes,
        uint32_t framesPerPoke,
        bool saveRaw)
        :   mp_fps(fps),
            m_numPokes(numPokes),
            m_framesPerPoke(framesPerPoke),
            m_framesPriorToPoke((uint32_t) (framesPerPoke * 0.1)),
            m_saveRaw(saveRaw)
{
    // Adopt WFS stream
    mp_IHwfs = ImageHandler2D<float>::newHandler2DAdoptImage(wfsStream->name);

    // Initialize Recording and evaluation image handlers

    // Initialize poke response sequence
    std::string pokeSeqName = m_streamPrefix;
    pokeSeqName += "pokeRespSeq";
    mp_IHpokeResponseSequence = ImageHandler2D<float>::newImageHandler2D(
        pokeSeqName,
        mp_IHwfs->mWidth,
        mp_IHwfs->mHeight,
        m_framesPerPoke * 2); // One set of frames prior to poke, one set after
    
    // Initialize poke response
    std::string pokeRespName = m_streamPrefix;
    pokeRespName += "pokeResp";
    mp_IHpokeResponse = ImageHandler2D<float>::newHandler2DfrmImage(
        pokeRespName, mp_IHwfs->getImage());

    // Initialize latency sequence
    std::string latSeqName = m_streamPrefix;
    latSeqName += "lateSeq";
    mp_IHLatencySequence = ImageHandler2D<float>::newImageHandler2D(
        latSeqName, mp_IHwfs->mWidth, m_framesPerPoke);

    // Initialize poke amplitude collection image
    std::string pokeAmpsName = m_streamPrefix;
    pokeAmpsName += "pokeAmps";
    mp_IHpokeAmps = ImageHandler2D<float>::newImageHandler2D(
        pokeAmpsName, m_framesPerPoke, m_numPokes);
    std::string pokeTimesName = m_streamPrefix;
    pokeTimesName += "pokeTimesRel";
    mp_IHpokeTimesRel = ImageHandler2D<double>::newImageHandler2D(
        pokeTimesName, m_framesPerPoke, m_numPokes);
}

double MLS_SequenceManager::measureFPS(float measurementTime_s)
{
    // Sync with WFS START
    mp_IHwfs->waitForNextFrame();
    auto t_start = high_resolution_clock::now();
    long wfscntstart = mp_IHwfs->getCnt0();

    // Wait for given duration
    std::this_thread::sleep_for(
        milliseconds((int) measurementTime_s * 1000));

    // Sync with WFS END
    mp_IHwfs->waitForNextFrame();
    auto t_end = high_resolution_clock::now();
    long wfscntend = mp_IHwfs->getCnt0();

    // Check if enough samples were collected
    if ( wfscntend - wfscntstart < 5)
        throw std::runtime_error(
            "Number of frames in time window too small -> cannot proceed.");
    else
    {   // Evaluate
        auto timeDelta = t_end - t_start;
        double timeDelta_ns = duration_cast<nanoseconds>(timeDelta).count();
        long numFrames = wfscntend - wfscntstart;

        return 1e9 * numFrames / timeDelta_ns;
    }
}

void MLS_SequenceManager::recordPokeResponse(bool postPoke)
{
    // PrePoke = first half of sequence, postPoke = second half
    // Both halves feature m_framesPerPoke frames.
    if (!postPoke)
        recordSequence(mp_IHpokeResponseSequence, m_framesPerPoke, 0);
    else
        recordSequence(mp_IHpokeResponseSequence, m_framesPerPoke, m_framesPerPoke);
}

void MLS_SequenceManager::evalPokeResponse()
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
    
    // subtract mean and normalize to single poke
    float mean = 0;
    for (int pxIdx = 0; pxIdx < pxPerFrame; pxIdx++)
        mean += dst[pxIdx];
    mean /= pxPerFrame;
    for (int pxIdx = 0; pxIdx < pxPerFrame; pxIdx++)
        dst[pxIdx] = (dst[pxIdx] - mean) / sequenceFrames/2;
    mp_IHpokeResponse->updateWrittenImage();
    // Save poke prior to RMS normalization
    std::string pokeFilename = "PokeResponse";
    mp_IHpokeResponse->saveToFPSdataDir(mp_fps, pokeFilename);

    // Normalize to RMS
    float rms = 0;
    float peak = 0;
    float valley = 0;
    for (int pxIdx = 0; pxIdx < pxPerFrame; pxIdx++)
    {
        peak = dst[pxIdx] > peak ? dst[pxIdx] : peak;
        valley = dst[pxIdx] < valley ? dst[pxIdx] : valley;
        rms += dst[pxIdx]*dst[pxIdx];
    }
    rms = sqrt(rms/pxPerFrame);
    for (int pxIdx = 0; pxIdx < pxPerFrame; pxIdx++)
        dst[pxIdx] /= rms;        
    mp_IHpokeResponse->updateWrittenImage();

    // Save statistics about poke
    float PtV = peak - valley;
    std::string pokeStatsOutputName = mp_fps->md->datadir;
    pokeStatsOutputName += "/" + pokeFilename + ".fits.stats";
    auto statsStream = std::ofstream(pokeStatsOutputName);
    statsStream << "pokeFilename=" << pokeFilename << "\n";
    statsStream << "PtV=" << PtV << "\n";
    statsStream << "RMS=" << rms << std::endl;
    statsStream.close();
}



_V2::system_clock::time_point MLS_SequenceManager::recordLatencyPokeSequence(
    uint32_t iteration, double interframePokeDelay_ns, bool postPoke)
{
    // == Preparations

    // Get the pointer location for the timestamps
    double* timestampDst = mp_IHpokeTimesRel->getWriteBuffer();
    // Skip to line of the current iteration
    timestampDst += mp_IHpokeTimesRel->mWidth * iteration;

    if (!postPoke)
    {   // == Record the pre-poke part of the sequence
        recordSequence(mp_IHLatencySequence,
            m_framesPriorToPoke, 0,
            timestampDst);

        // Create the time point at which the poke shall happen,
        // based on the aquisition time of the last frame.
        double pokeTime_ns = timestampDst[m_framesPriorToPoke - 1]
                             + interframePokeDelay_ns;
        std::chrono::nanoseconds ns((long long) pokeTime_ns);
        return std::chrono::system_clock::time_point(ns);
    }
    else
    {   // Record the post-poke part of the sequence
        recordSequence(mp_IHLatencySequence,
            m_framesPerPoke - m_framesPriorToPoke, m_framesPriorToPoke,
            timestampDst);
        mp_IHLatencySequence->updateWrittenImage();

        if (m_saveRaw)
        {   // Save the raw sequence cube
            char* iterString = new char[10];
            sprintf(iterString, "%04d", iteration);
            std::string fname = "mlat-testC-";
            fname += iterString;

            mp_IHLatencySequence->saveToFPSdataDir(mp_fps, fname);
            delete[] iterString;
        }

        // Subtract the poke time from the timestamps
        double prePokeFrameTimestamp_ns = timestampDst[m_framesPriorToPoke - 1];
        for (int i = 0; i < m_framesPerPoke; i++)
        {
            timestampDst[i] -= prePokeFrameTimestamp_ns;
            timestampDst[i] -=  interframePokeDelay_ns;
        }
        mp_IHpokeTimesRel->updateWrittenImage();

        return std::chrono::high_resolution_clock::now();
    }
}

std::pair<double*, float*> MLS_SequenceManager::decomposeLastPokeSequence(
    uint32_t iteration)
{
    // == Get array pointers
    float* responseMode = mp_IHpokeResponse->getWriteBuffer();
    float* sequence = mp_IHLatencySequence->getWriteBuffer();
    float* amplitudeDst = mp_IHpokeAmps->getWriteBuffer();
    amplitudeDst += mp_IHpokeAmps->mWidth * iteration;
    double* timeDst = mp_IHpokeTimesRel->getWriteBuffer();
    timeDst += mp_IHpokeTimesRel->mWidth * iteration;

    // == Calculate baseline
    int pxPerFrame = mp_IHLatencySequence->mWidth;
    // Init baseline array
    float* baseline = new float[pxPerFrame];
    for (int px = 0; px < pxPerFrame; px++)
        baseline[px] = 0;
    // Add up frames prior to poke
    for (int frame = 0; frame < m_framesPriorToPoke; frame++)
        for (int px = 0; px < pxPerFrame; px++)
            baseline[px] += sequence[frame * pxPerFrame + px];
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
        // Extract the response amplitude from the current frame
        amplitude = 0;
        for (int px = 0; px < pxPerFrame; px++)
        {
            curWFSval = sequence[frame * pxPerFrame + px];
            amplitude +=  (curWFSval - baseline[px]) * responseMode[px];
        }
        amplitudeDst[frame] = amplitude;
        
        // Average the last frames for normalization
        if (frame >= 0.9*m_framesPerPoke)
        {
            avgSettleValue += amplitude;
            settleValueFrames++;
        }
    }
    avgSettleValue /= settleValueFrames;
    // == Normalize to last values
    for (int frame = 0; frame < m_framesPerPoke; frame++)
        amplitudeDst[frame] /= avgSettleValue;
    
    // == Cleanup
    delete[] baseline;

    return {timeDst, amplitudeDst};
}



double MLS_SequenceManager::calcRawSettlingLatency()
{
    double hwLatencyRaw_us = 0;
    for (int seqIdx = 0; seqIdx < mp_IHpokeAmps->mHeight; seqIdx++)
    {
        double sequenceSettlingTime = 0;
        for (int smplIdx = 0; smplIdx < mp_IHpokeAmps->mWidth; smplIdx++)
        {
            float amp = mp_IHpokeAmps->read(smplIdx, seqIdx);
        if (amp < 0.9 || amp > 1.1)
            sequenceSettlingTime = mp_IHpokeTimesRel->read(smplIdx, seqIdx);
        }
        hwLatencyRaw_us += sequenceSettlingTime / 1000;
    }
    hwLatencyRaw_us /= mp_IHpokeAmps->mHeight;
    return hwLatencyRaw_us;
}



void MLS_SequenceManager::smoothLatencyDecomposition(
    std::shared_ptr<MLS_ResultManager> p_resultMngr)
{
    std::vector<std::pair<double, float>> samples;
    samples = getTimeSortedResponseSamples();

    // == Span time grid over recorded window
    double timeResolution_ns = p_resultMngr->getWfsDt_us() * 1000 / 10;  // 10 samples per frame
    double minTime_ns = samples[0].first;
    double maxTime_ns = samples[samples.size() - 1].first;
    int numSamples = ceil((maxTime_ns - minTime_ns) / timeResolution_ns);

    // Initialize smoothed time amplitude image
    std::string smoothedTimeName = m_streamPrefix;
    smoothedTimeName += "smoothedPokeTimes";
    mp_IHpokeTimeSmoothed = ImageHandler2D<double>::newImageHandler2D(
        smoothedTimeName, numSamples, 1);
    // Initialize smoothed poke amplitude image
    std::string smoothedAmpName = m_streamPrefix;
    smoothedAmpName += "smoothedPokeAmps";
    mp_IHpokeAmpSmoothed = ImageHandler2D<float>::newImageHandler2D(
        smoothedAmpName, numSamples, 1);
    // Initialize smoothed poke stddev image
    std::string smoothedStddevName = m_streamPrefix;
    smoothedStddevName += "smoothedPokeStdDev";
    mp_IHpokeStdDevSmoothed = ImageHandler2D<float>::newImageHandler2D(
        smoothedStddevName, numSamples, 1);
    
    // == Convolve sequence with gaussian kernel
    double kernelStdDev_ns = p_resultMngr->getWfsDt_us() * 1000 / 4;
    double kernelDenominator = -2*kernelStdDev_ns*kernelStdDev_ns;

    p_resultMngr->setSmoothingProperties(
        timeResolution_ns/1000, kernelStdDev_ns/1000);
    
    for (int i = 0; i < numSamples; i++)
    {
        double kernelCenterTime_ns = minTime_ns + i * timeResolution_ns;

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
        mp_IHpokeTimeSmoothed->write(kernelCenterTime_ns / 1000., i, 0);
        mp_IHpokeAmpSmoothed->write(integral / kernelSum, i, 0);
    }
    mp_IHpokeTimeSmoothed->updateWrittenImage();
    mp_IHpokeAmpSmoothed->updateWrittenImage();

    // Calculate the smoothed standard deviation    
    for (int i = 0; i < numSamples; i++)
    {
        double kernelCenterTime_ns = mp_IHpokeTimeSmoothed->read(i, 0) * 1000;

        double kernelSum = 0;
        double integral = 0;
        
        for (int k = 0; k < samples.size(); k++)
        {
            std::pair<double, float> sample = samples.at(k);
            double delT = sample.first - kernelCenterTime_ns;
            double kernelValue = exp(delT*delT/kernelDenominator);

            // Interpolate smoothed value to sample time
            double kFloat = (sample.first - minTime_ns) / timeResolution_ns;
            int kLower = floor(kFloat);
            if (kLower < 0) kLower = 0;
            if (kLower >= numSamples - 1) kLower = numSamples - 2;
            kFloat -= kLower;
            float valLower = mp_IHpokeAmpSmoothed->read(kLower, 0);
            float valUpper = mp_IHpokeAmpSmoothed->read(kLower + 1, 0);
            double valCenter = valLower * (1-kFloat) + valUpper * kFloat;
            double deviation = sample.second - valCenter;

            kernelSum += kernelValue;
            integral += deviation*deviation * kernelValue;
        }
        // Write output to image
        double stdDev = sqrt(integral / kernelSum);
        mp_IHpokeStdDevSmoothed->write(stdDev, i, 0);
        // Write output to file
        p_resultMngr->logSmoothedAmplitude(
            mp_IHpokeTimeSmoothed->read(i, 0),
            mp_IHpokeAmpSmoothed->read(i, 0),
            stdDev);
    }
    mp_IHpokeStdDevSmoothed->updateWrittenImage();
}

double MLS_SequenceManager::calcHWdelay()
{
    double t_10 = 0;
    double t_70 = 0;
    for (int i = 0; i < mp_IHpokeAmpSmoothed->mWidth; i++)
    {
        double time = mp_IHpokeTimeSmoothed->read(i, 0);
        float amp = mp_IHpokeAmpSmoothed->read(i, 0);
        if (amp < 0.1)
            t_10 = time;
        if (amp < 0.7)
            t_70 = time;
    }
    // Assuming a linear rise time calculate beginning of movement
    return (t_10 - (t_70 - t_10) / 6);
}



double MLS_SequenceManager::calcRiseime()
{
    double t_10 = 0;
    double t_90_110 = 0;
    for (int i = 0; i < mp_IHpokeAmpSmoothed->mWidth; i++)
    {
        double time = mp_IHpokeTimeSmoothed->read(i, 0);
        float amp = mp_IHpokeAmpSmoothed->read(i, 0);
        if (amp < 0.1)
            t_10 = time;
        if (amp < 0.9 || amp > 1.1)
            t_90_110 = time;
    }
    // Calc 10-90% rise time, including setteling between 90%-110%
    return (t_90_110 - t_10);
}



double MLS_SequenceManager::calcHWlatency()
{
    double t_90_110 = 0;
    for (int i = 0; i < mp_IHpokeAmpSmoothed->mWidth; i++)
    {
        double time = mp_IHpokeTimeSmoothed->read(i, 0);
        float amp = mp_IHpokeAmpSmoothed->read(i, 0);
        if (amp < 0.9 || amp > 1.1)
            t_90_110 = time;
    }
    // Calc hw latency
    return t_90_110;
}



void MLS_SequenceManager::waitNumFrames(uint32_t numFrames)
{
    for (int i = 0; i < numFrames; i++)
        mp_IHwfs->waitForNextFrame();
}



void MLS_SequenceManager::recordSequence(
        spImHandler2D(float) IHdst,
        int numFrames,
        int dstFrameOffset,
        double* timestampDst)
{
    bool recTime = timestampDst != nullptr;
    // Get raw data pointers
    float* src = mp_IHwfs->getWriteBuffer();
    float* dst = IHdst->getWriteBuffer();
    double* dstT = timestampDst;
    // Apply frame offset
    dst += mp_IHwfs->mNumPx * dstFrameOffset;
    if (recTime)
        dstT += dstFrameOffset;
    // Record the sequence
    for (int i = 0; i < numFrames; i++)
    {
        mp_IHwfs->waitForNextFrame();
        // Record time
        if (recTime)
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto nanoseconds_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
            dstT[i] = (double) nanoseconds_since_epoch.count();
        }
        // Copy pixels
        for (int k = 0; k < mp_IHwfs->mNumPx; k++)
            dst[k] = src[k];
        // Proceed to next frame
        dst += mp_IHwfs->mNumPx;
    }
}


std::vector<std::pair<double, float>> MLS_SequenceManager::getTimeSortedResponseSamples()
{
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

    return samples;
}
