#ifndef POINT_HPP
#define POINT_HPP

#include <cmath>
#include <cstdio>
#include <typeinfo>

template<typename T>
class Point
{
public:
    T mX = 0;
    T mY = 0;
    float mIntensity = 1.;
    bool mValid = true;

    Point();
    Point(T x, T y, float intensity, bool valid = true);
    Point(T x, T y, bool valid = true);

    double abs() const { return sqrt(mX*mX + mY*mY); }
    static double abs(Point<T> const &P) { return P.abs(); }
    template<typename U>
    double distance(Point<U> const &P)
        { return abs(((Point<double>)*this) - P); }
    void print();

    // Operators
    template <typename U>
    Point<T> operator + (Point<U> const &addend);
    template <typename U>
    void operator += (Point<U> const &addend);
    template <typename U>
    Point<T> operator + (U const &addend);
    template <typename U>
    void operator += (U const &addend);

    template <typename U>
    Point<T> operator - (Point<U> const &subtrahend);
    template <typename U>
    void operator -= (Point<U> const &subtrahend);
    template <typename U>
    Point<T> operator - (U const &subtrahend);
    template <typename U>
    void operator -= (U const &subtrahend);

    template <typename U>
    Point<T> operator * (Point<U> const &factor);
    template <typename U>
    void operator *= (Point<U> const &factor);
    template <typename U>
    Point<T> operator * (U const &factor);
    template <typename U>
    void operator *= (U const &factor);

    template <typename U>
    Point<T> operator / (Point<U> const &divisor);
    template <typename U>
    void operator /= (Point<U> const &divisor);
    template <typename U>
    Point<T> operator / (U const &divisor);
    template <typename U>
    void operator /= (U const &divisor);

    template <typename U>
    Point<T> operator % (Point<U> const &divisor);
    template <typename U>
    Point<T> operator % (U const &divisor);

    template <typename U>
    void operator = (Point<U> const &P);
    template <typename U>
    operator Point<U>() { return Point<U>(mX, mY, mIntensity, mValid); };

    template <typename U>
    bool operator == (Point<U> const &P);
    template <typename U>
    bool operator != (Point<U> const &P);

};

// ========== CONSTRUCTORS ==========
template <typename T>
inline Point<T>::Point()
{}

template <typename T>
inline Point<T>::Point(T x, T y, float intensity, bool valid)
    : mX(x), mY(y), mIntensity(intensity), mValid(valid)
{}

template <typename T>
inline Point<T>::Point(T x, T y, bool valid)
    : mX(x), mY(y), mValid(valid)
{}

// ========== UTILITY ==========
template <typename T>
void Point<T>::print()
{
    printf("Point type=%s, X=%s; Y=%s; Intensity=%s; Valid=%s\n",
        typeid(T).name(),
        std::to_string(mX).c_str(),
        std::to_string(mY).c_str(),
        std::to_string(mIntensity).c_str(),
        mValid ? "true" : "false");
}

// ========== OPERATORS ==========
// ### Addition ###
template <typename T>
template <typename U>
inline Point<T> Point<T>::operator + (Point<U> const &addend)
{
    Point<T> sum;
    sum.mX = mX + (T) addend.mX;
    sum.mY = mY + (T) addend.mY;
    sum.mIntensity = mIntensity + addend.mIntensity;
    sum.mValid = mValid && addend.mValid;
    return sum;
}

template <typename T>
template <typename U>
inline void Point<T>::operator += (Point<U> const &addend)
{
    mX += (T) addend.mX;
    mY += (T) addend.mY;
    mIntensity += addend.mIntensity;
    mValid = mValid && addend.mValid;
}

template <typename T>
template <typename U>
inline Point<T> Point<T>::operator + (U const &addend)
 {
    Point<T> sum;
    sum.mX = mX + (T) addend;
    sum.mY = mY + (T) addend;
    sum.mIntensity = mIntensity;
    sum.mValid = mValid;
    return sum;
}

template <typename T>
template <typename U>
inline void Point<T>::operator += (U const &addend)
 {
    mX += (T) addend;
    mY += (T) addend;
}

// ### Subraction ###
template <typename T>
template <typename U>
inline Point<T> Point<T>::operator - (Point<U> const &subtrahend)
{
    Point<T> dif;
    dif.mX = mX - (T) subtrahend.mX;
    dif.mY = mY - (T) subtrahend.mY;
    dif.mIntensity = mIntensity - subtrahend.mIntensity;
    dif.mValid = mValid && subtrahend.mValid;
    return dif;
}

template <typename T>
template <typename U>
inline void Point<T>::operator -= (Point<U> const &subtrahend)
{
    mX -= (T) subtrahend.mX;
    mY -= (T) subtrahend.mY;
    mIntensity -= subtrahend.mIntensity;
    mValid = mValid && subtrahend.mValid;
}

template <typename T>
template <typename U>
inline Point<T> Point<T>::operator - (U const &subtrahend)
 {
    Point<T> dif;
    dif.mX = mX - (T) subtrahend;
    dif.mY = mY - (T) subtrahend;
    dif.mIntensity = mIntensity;
    dif.mValid = mValid;
    return dif;
}

template <typename T>
template <typename U>
inline void Point<T>::operator -= (U const &subtrahend)
 {
    mX -= (T) subtrahend;
    mY -= (T) subtrahend;
}

// ### Multiplication ###
template <typename T>
template <typename U>
inline Point<T> Point<T>::operator * (Point<U> const &factor)
{
    Point<T> prod;
    prod.mX = mX * (T) factor.mX;
    prod.mY = mY * (T) factor.mY;
    prod.mIntensity = mIntensity * factor.mIntensity;
    prod.mValid = mValid && factor.mValid;
    return prod;
}

template <typename T>
template <typename U>
inline void Point<T>::operator *= (Point<U> const &factor)
{
    mX *= (T) factor.mX;
    mY *= (T) factor.mY;
    mIntensity *= factor.mIntensity;
    mValid = mValid && factor.mValid;
}

template <typename T>
template <typename U>
inline Point<T> Point<T>::operator * (U const &factor)
 {
    Point<T> prod;
    prod.mX = mX * (T) factor;
    prod.mY = mY * (T) factor;
    prod.mIntensity = mIntensity;
    prod.mValid = mValid;
    return prod;
}

template <typename T>
template <typename U>
inline void Point<T>::operator *= (U const &factor)
 {
    mX *= (T) factor;
    mY *= (T) factor;
}

// ### Division ###
template <typename T>
template <typename U>
inline Point<T> Point<T>::operator / (Point<U> const &divisor)
{
    Point<T> quo;
    quo.mX = mX / (T) divisor.mX;
    quo.mY = mY / (T) divisor.mY;
    quo.mIntensity = mIntensity / divisor.mIntensity;
    quo.mValid = mValid && divisor.mValid;
    if (divisor.mX == 0 || divisor.mY == 0 || divisor.mIntensity == 0)
        quo.mValid = false;
    return quo;
}

template <typename T>
template <typename U>
inline void Point<T>::operator /= (Point<U> const &divisor)
{
    mX /= (T) divisor.mX;
    mY /= (T) divisor.mY;
    mIntensity /= divisor.mIntensity;
    mValid = mValid && divisor.mValid;
    if (divisor.mX == 0 || divisor.mY == 0 || divisor.mIntensity == 0)
        mValid = false;
}

template <typename T>
template <typename U>
inline Point<T> Point<T>::operator / (U const &divisor)
 {
    Point<T> quo;
    quo.mX = mX / (T) divisor;
    quo.mY = mY / (T) divisor;
    quo.mIntensity = mIntensity;
    quo.mValid = mValid && (divisor != 0);
    return quo;
}

template <typename T>
template <typename U>
inline void Point<T>::operator /= (U const &divisor)
 {
    mX /= (T) divisor;
    mY /= (T) divisor;
    mValid = mValid && (divisor != 0);
}

// ### Modulo ###
template <typename T>
template <typename U>
inline Point<T> Point<T>::operator % (Point<U> const &divisor)
{
    Point<T> mod;
    mod.mX = mX - divisor.mX * floor(mX / divisor.mX);
    mod.mY = mY - divisor.mY * floor(mY / divisor.mY);
    mod.mIntensity = mIntensity - divisor.mIntensity * floor(mIntensity / divisor.mIntensity);
    mod.mValid = mValid && divisor.mValid;
    if (divisor.mX == 0 || divisor.mY == 0 || divisor.mIntensity == 0)
        mod.mValid = false;
    return mod;
}

template <typename T>
template <typename U>
inline Point<T> Point<T>::operator % (U const &divisor)
 {
    Point<T> mod;
    mod.mX = mX - divisor * floor(mX / divisor);
    mod.mY = mY - divisor * floor(mY / divisor);
    mod.mIntensity = mIntensity;
    mod.mValid = mValid && (divisor != 0);
    return mod;
}

// ### Assignment ###
template <typename T>
template <typename U>
inline void Point<T>::operator = (Point<U> const &P)
{
    mX = (T) P.mX;
    mY = (T) P.mY;
    mIntensity = P.mIntensity;
    mValid = P.mValid;
}

// ### Comparison ###
// Attention: Converstion towards left type prior to conversion.
// E.g. if:
// - Left point is or type int
// - Right point is of type double
// -> double values will be rounded towards int and then compared.
template <typename T>
template <typename U>
inline bool Point<T>::operator == (Point<U> const &P)
{
    return
        (mX == (T) P.mX) &&
        (mY == (T) P.mY) &&
        (mIntensity == P.mIntensity) &&
        (mValid == P.mValid);
}

template <typename T>
template <typename U>
inline bool Point<T>::operator != (Point<U> const &P)
{
    return ! (*this == P);
}


#endif // POINT_HPP