// Utility for dealing with the data types used by ISIO

#ifndef ATYPEUTIL_HPP
#define ATYPEUTIL_HPP

#include <stdint.h>
#include "ImageStreamIO/ImageStreamIO.h"

// Conversion for foreign element access
template <typename T>
T convertAtypeArrayElement(void* arrptr, uint8_t arrayAtype, uint32_t index)
{
    switch(arrayAtype) {
        case _DATATYPE_UINT8: return (T)(*(((uint8_t*)arrptr)+index));
        case _DATATYPE_INT8: return (T)(*(((int8_t*)arrptr)+index));
        case _DATATYPE_UINT16: return (T)(*(((uint16_t*)arrptr)+index));
        case _DATATYPE_INT16: return (T)(*(((int16_t*)arrptr)+index));
        case _DATATYPE_UINT32: return (T)(*(((uint32_t*)arrptr)+index));
        case _DATATYPE_INT32: return (T)(*(((int32_t*)arrptr)+index));
        case _DATATYPE_UINT64: return (T)(*(((uint64_t*)arrptr)+index));
        case _DATATYPE_INT64: return (T)(*(((int64_t*)arrptr)+index));
        case _DATATYPE_FLOAT: return (T)(*(((float*)arrptr)+index));
        case _DATATYPE_DOUBLE: return (T)(*(((double*)arrptr)+index));
        default:
        throw std::runtime_error("convertAtypeArrayElement<T>: No case for this data type.\n");
    }
}

// Get the atype from the generic type
template <typename T>
uint8_t getAtype()
{
    throw std::runtime_error("getAtype<T>: No specification for this data type.\n");
}

// Get the atype from the generic type - specializations
#define GETATYPE(type, returnVal) template <> inline uint8_t getAtype<type>() { return returnVal; }
GETATYPE(uint8_t, _DATATYPE_UINT8)
GETATYPE(int8_t, _DATATYPE_INT8)
GETATYPE(uint16_t, _DATATYPE_UINT16)
GETATYPE(int16_t, _DATATYPE_INT16)
GETATYPE(uint32_t, _DATATYPE_UINT32)
GETATYPE(int32_t, _DATATYPE_INT32)
GETATYPE(uint64_t, _DATATYPE_UINT64)
GETATYPE(int64_t, _DATATYPE_INT64)
GETATYPE(float, _DATATYPE_FLOAT)
GETATYPE(double, _DATATYPE_DOUBLE)

// Returns true if the atype matches the generic type
template <typename T>
bool checkAtype(uint8_t atype)
{
    uint8_t genericAtype;
    try {
        genericAtype = getAtype<T>();
    }
    catch (std::runtime_error err)
    {
        throw std::runtime_error("chackAtype<T>: no implementation of getAtype<T> for this type.");
    }
    return genericAtype == atype;
}

#endif // ATYPEUTIL_HPP