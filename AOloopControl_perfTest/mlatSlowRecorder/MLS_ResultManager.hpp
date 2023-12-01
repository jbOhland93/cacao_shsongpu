#ifndef MLS_RESULTMANAGER_HPP
#define MLS_RESULTMANAGER_HPP

#include "../../AOloopControl_IOtools/shs_on_gpu/util/ImageHandler2D.hpp"
#include "MLS_PokePattern.hpp"
#include <fstream>

// A class for managing the results obtained during latency recording
class MLS_ResultManager
{
public:
    MLS_ResultManager(
        FUNCTION_PARAMETER_STRUCT* fps, // process related fps
        PokePattern pokePattern,// Poke pattern
        float maxActStroke,     // Maximum actuator stroke in pattern
        uint32_t numPokes,      // number of iterations
        uint32_t framesPerPoke);// number of frames per iteration
    ~MLS_ResultManager();

    // Result setters
    void setFPS(double FPS_Hz);
    void logRawAmplitude(double* times, float* amplitudes);
    void logSmoothedAmplitude(double time, float amplitude);
    void setHWdelay(double hwDelay_us);
    void setRisetime(double risetime_us);
    void setHwLatencyRaw(double latency_us);
    void setHwLatencySmoothed(double latency_us);

    // Result getters
    double getWfsDt_us() { return m_wfsdt_us; };

    // Write properties of smoothing properties to output file
    // resolution_us - time resolution of the smoothing in us
    // stdDev_us - the stdDev of the employed gaussian kernel
    void setSmoothingProperties(double resolution_us, double stdDev_us);

    // Publish all numbers in fps and on disk.
    // Also, print some output for the user.
    void publishResults();

private:
    // Parameters
    FUNCTION_PARAMETER_STRUCT* mp_fps;
    float m_fpsMeasurementTime;
    PokePattern m_pokePattern;
    float m_maxStroke;
    uint32_t m_numPokes;
    uint32_t m_framesPerPoke;
    uint32_t m_framesPriorToPoke;

    // Output files
    std::ofstream m_cacaovarsOutput;
    std::ofstream m_pokeAmpOutput;
    std::ofstream m_smoothedAmpOutput;

    // Intermediate results
    double m_wfsdt_us;
    double m_FPS_Hz;

    // == RESULTS ==
    // Raw results
    // Here, no floating average is applied.
    // Therefore, the latency correspond to the poke with the slowest settling time.
    // These should be used as a conservative assumption if stability is more
    // important than speed.
    float m_hwLatencyRaw_frames;    // Delay from poke to [90%|110%] stroke
    float m_hwLatencyRaw_us;        // Delay from poke to [90%|110%] of stroke
    // Smoothed results
    // Here, a floating average over all pokes is taken for the evaluation
    // Therefore, uncorrelated ringing cancles out.
    float m_hwDelay_frames;         // Delay from poke to first movement
    float m_hwDelay_us;             // Delay from poke to first movement
    float m_riseTime10to90_us;      // Rise time from 10% to 90% of stroke
    float m_riseTime10to90_frames;  // Rise time from 10% to 90% of stroke
    float m_hwLatency_frames;       // Delay from poke to [90%|110%] stroke
    float m_hwLatency_us;           // Delay from poke to [90%|110%] of stroke

    // Sets an FPS value, saves it to disk and provides a message to the user
    void setsavelog_fpsvalFloat(
                std::string name,
                std::string msg,
                float val,
                std::string cacaoVarName = "");
};

#endif // MLS_RESULTMANAGER_HPP
