#ifndef MLS_POKEPATTERN_H
#define MLS_POKEPATTERN_H
#include <stdint.h>

enum class PokePattern : int32_t {
    SHMIM = -1,
    HOMOGENEOUS = 0,
    SINE = 1,
    CHECKERBOARD = 2,
    SQUARE = 3,
    HALFSQUARE = 4,
    DOUBLESQUARE = 5,
    XRAMP = 6,
    XHALF = 7,
    YRAMP = 8,
    YHALF = 9,
};

#endif // MLS_POKEPATTERN_H
