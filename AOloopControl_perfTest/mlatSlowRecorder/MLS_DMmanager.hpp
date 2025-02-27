#ifndef MLS_DMMANAGER_HPP
#define MLS_DMMANAGER_HPP

#include "../../AOloopControl_IOtools/shs_on_gpu/util/ImageHandler2D.hpp"
#include "MLS_PokePattern.hpp"

// A class managing a DM input stream
// with the ability to apply alternaing poke patterns
class MLS_DMmanager
{
public:
    MLS_DMmanager(
        IMAGE* dmstream,            // Stream of the DM input
        PokePattern pokePattern,    // Poke pattern
        std::string patternImage,   // Patter image name
        uint32_t shmPatternIdx,     // Index of shmIm pattern slice
        float patternToStrokeMul);  // Pattern to stroke factor

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
    
private:
    float m_strokeMul;
    PokePattern m_pokePattern;

    // DM input stream
    spImHandler2D(float) mp_IHdm;
    // ShmIm poke patterns
    spImHandler2D(float) mp_IHpatterns;
    // Poke streams
    spImHandler2D(float) mp_IHdmPoke0;
    spImHandler2D(float) mp_IHdmPoke1;
};

#endif // MLS_DMMANAGER_HPP