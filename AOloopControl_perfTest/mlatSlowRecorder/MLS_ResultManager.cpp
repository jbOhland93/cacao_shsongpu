#include "MLS_ResultManager.hpp"
#include <iostream>

MLS_ResultManager::MLS_ResultManager(
        FUNCTION_PARAMETER_STRUCT* fps, // process related fps
        PokePattern pokePattern,// Poke pattern:
        float maxActStroke,     // Maximum actuator stroke in pattern
        uint32_t numPokes,      // number of iterations
        uint32_t framesPerPoke) // number of frames per iteration
        :   mp_fps(fps),
            m_pokePattern(pokePattern),
            m_maxStroke(maxActStroke),
            m_numPokes(numPokes),
            m_framesPerPoke(framesPerPoke),
            m_framesPriorToPoke((uint32_t) (framesPerPoke * 0.1))
{
    // Create output files
    std::string cacaovarsName = mp_fps->md->datadir;
    cacaovarsName += "/cacaovars.bash";
    m_cacaovarsOutput = std::ofstream(cacaovarsName);
    if (!m_cacaovarsOutput.is_open())
        throw std::runtime_error(cacaovarsName + "could not be opened.");

    std::string pokeAmpOutputName = mp_fps->md->datadir;
    pokeAmpOutputName += "/pokeAmpOutput.dat";
    m_pokeAmpOutput = std::ofstream(pokeAmpOutputName);
    if (!m_pokeAmpOutput.is_open())
        throw std::runtime_error(pokeAmpOutputName + "could not be opened.");
    m_pokeAmpOutput << "# Responses of each individual iteration\n";
    m_pokeAmpOutput << "# \n";
    m_pokeAmpOutput << "# 1: frame index, 0=first frame after poke\n";
    m_pokeAmpOutput << "# 2: time in us relative to poke time\n";
    m_pokeAmpOutput << "# 3: normalized response\n";
    m_pokeAmpOutput << "# Note: Each sequence is followed by a line containing information\n"
                    << " about the frame that marks the last one not being conained in the\n"
                    << " [.9|1.1] inverval:\n";
    m_pokeAmpOutput << "# \t #\tFrameIndexOf0.9Response\tTimeOf0.9ResponseInSeconds\n";
    m_pokeAmpOutput << "# Max actuator stroke = " << m_maxStroke << "\n";
    m_pokeAmpOutput << "# Poke pattern = " << (int32_t) m_pokePattern << "\n";

    std::string smoothedAmpOutputName = mp_fps->md->datadir;
    smoothedAmpOutputName += "/smoothedPokeAmps.dat";
    m_smoothedAmpOutput = std::ofstream(smoothedAmpOutputName);
    if (!m_smoothedAmpOutput.is_open())
        throw std::runtime_error(smoothedAmpOutputName + "could not be opened.");
    m_smoothedAmpOutput << "# Temporal averaging of all respones\n";
    m_smoothedAmpOutput << "# Max actuator stroke = " << m_maxStroke << "\n";
    m_smoothedAmpOutput << "# Poke pattern = " << (int32_t) m_pokePattern << "\n";
}

MLS_ResultManager::~MLS_ResultManager()
{
    m_cacaovarsOutput.close();
    m_pokeAmpOutput.close();
    m_smoothedAmpOutput.close();
}

void MLS_ResultManager::setFPS(double FPS_Hz)
{
    m_FPS_Hz = FPS_Hz;
    m_wfsdt_us = 1e6/FPS_Hz;
    
    // Write to files
    m_cacaovarsOutput << "export CACAO_WFSFRATE=" << m_FPS_Hz << "\n";
    m_pokeAmpOutput << "# WFS dt: " << m_wfsdt_us << "us (FPS=" << m_FPS_Hz << "Hz)\n";
    m_smoothedAmpOutput << "# WFS dt: " << m_wfsdt_us << "us (FPS=" << m_FPS_Hz << "Hz)\n";

    // Print notice
    std::cout << "\t wfs dt = " << m_wfsdt_us << "us "
        << "(" << m_FPS_Hz << "Hz)\n";
}

void MLS_ResultManager::logRawAmplitude(double* times_ns, float* amplitudes)
{
    long i_90 = 0;
    double t_90 = 0;
    for (int frame = 0; frame < m_framesPerPoke; frame++)
    {
        m_pokeAmpOutput << frame - (int)m_framesPriorToPoke << "\t";
        m_pokeAmpOutput << times_ns[frame]/1000. << "\t";
        m_pokeAmpOutput << amplitudes[frame] << "\n";
        if (amplitudes[frame] < 0.9)
        {
            i_90 = frame;
            t_90 = times_ns[frame]/1000.;
        }
    }
    m_pokeAmpOutput << "#\t" << i_90 << "\t" << t_90/1e6 << "\n";
}

void MLS_ResultManager::logSmoothedAmplitude(double time, float amplitude, float stdDev)
{
    m_smoothedAmpOutput << time << "\t" << amplitude << "\t" << stdDev << "\n";
}

void MLS_ResultManager::setHWdelay(double hwDelay_us)
{
    m_hwDelay_us = hwDelay_us;
    m_hwDelay_frames = hwDelay_us / m_wfsdt_us;
}
void MLS_ResultManager::setRisetime(double risetime_us)
{
    m_riseTime10to90_us = risetime_us;
    m_riseTime10to90_frames = risetime_us / m_wfsdt_us;
}

void MLS_ResultManager::setHwLatencyRaw(double latency_us)
{
    m_hwLatencyRaw_us = latency_us;
    m_hwLatencyRaw_frames = latency_us / m_wfsdt_us;

    std::cout   << "\t Raw settling time = " << m_hwLatencyRaw_us
                << "us (" << m_hwLatencyRaw_frames
                << " frames)\n";
}

void MLS_ResultManager::setHwLatencySmoothed(double latency_us)
{
    m_hwLatency_us = latency_us;
    m_hwLatency_frames = latency_us / m_wfsdt_us;
}

void MLS_ResultManager::setSmoothingProperties(
            double resolution_us, double stdDev_us)
{
    std::cout   << "\t Time grid sample spacing: "
                << resolution_us << "us (1/10 frame spacing) \n";
    std::cout   << "\t Using gaussian kernel with a stddev of "
                << stdDev_us << "us (1/4 frame spacing)\n";

    m_smoothedAmpOutput << "# \n";
    m_smoothedAmpOutput << "# Temporal resolution: "
                        << resolution_us
                        << "us (1/10 frame spacing)\n";
    m_smoothedAmpOutput << "# Temporal smoothing stdDev (gaussian): "
                        << stdDev_us
                        << "us (1/4 frame spacing)\n";
    m_smoothedAmpOutput << "# \n";
    m_smoothedAmpOutput << "# 1: Time relative to poke command in us\n";
    m_smoothedAmpOutput << "# 2: Normalized response\n";
    m_smoothedAmpOutput << "# 3: Standard deviation of normalized response\n";
    m_smoothedAmpOutput << "# \n";
}

void MLS_ResultManager::publishResults()
{
    setsavelog_fpsvalFloat(
        ".out.framerateHz",
        "Framerate (Hz)",
        m_FPS_Hz);
    setsavelog_fpsvalFloat(
        ".out.latency_fr",
        "Hardware latency smoothed (frames)",
        m_hwLatency_frames,
        "CACAO_LATENCYHARDWFR");
    setsavelog_fpsvalFloat(
        ".out.latency_us",
        "Hardware latency smoothed (us)",
        m_hwLatency_us,
        "CACAO_LATENCYHARDWUS");
    setsavelog_fpsvalFloat(
        ".out.latencyRaw_fr",
        "Hardware latency raw (frames)",
        m_hwLatencyRaw_frames,
        "CACAO_LATENCYHARDWRAWFR");
    setsavelog_fpsvalFloat(
        ".out.latencyRaw_us",
        "Hardware latency raw (us)",
        m_hwLatencyRaw_us,
        "CACAO_LATENCYHARDWRAWUS");
    setsavelog_fpsvalFloat(
        ".out.delay_fr",
        "Latency to first motion (frames)",
        m_hwDelay_frames,
        "CACAO_DELAYHARDWFR");
    setsavelog_fpsvalFloat(
        ".out.delay_us",
        "Latency to first motion (us)",
        m_hwDelay_us,
        "CACAO_DELAYHARDWUS");
    setsavelog_fpsvalFloat(
        ".out.risetime_fr",
        "Rise time (10\% to [90\%|110\%]) (frames)",
        m_riseTime10to90_frames,
        "CACAO_DMRISETIMEFR");
    setsavelog_fpsvalFloat(
        ".out.risetime_us",
        "Rise time (10\% to [90\%|110\%]) (us)",
        m_riseTime10to90_us,
        "CACAO_DMRISETIMEUS");
}

void MLS_ResultManager::setsavelog_fpsvalFloat(
        std::string name,
        std::string msg,
        float val,
        std::string cacaoVarName)
{
    functionparameter_SetParamValue_FLOAT32(mp_fps, name.c_str(), val);
    functionparameter_SaveParam2disk(mp_fps, name.c_str());
    if (!cacaoVarName.empty())
        m_cacaovarsOutput   << "export " << cacaoVarName << "="
                            << val << "\n";
    std::cout << "\t" << msg << " = " << val << "\n";
}
