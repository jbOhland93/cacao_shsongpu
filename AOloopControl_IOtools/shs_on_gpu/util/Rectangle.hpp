#ifndef RECTANGLE_HPP
#define RECTANGLE_HPP

#include "Point.hpp"

template<typename T>
class Rectangle
{
public:
    Point<T> mRoot = Point<T>(0,0,0,true);
    Point<T> mSize = Point<T>(0,0,0,true);
    bool mValid = true;

    Rectangle();
    Rectangle(T x, T y, T w, T h, bool valid = true);
    Rectangle(T x, T y, Point<T> s, bool valid = true);
    Rectangle(Point<T> r, T w, T h, bool valid = true);
    Rectangle(Point<T> r, Point<T> s, bool valid = true);

    T x() { return mRoot.mX; }
    T y() { return mRoot.mY; }
    T w() { return mSize.mX; }
    T h() { return mSize.mY; }

    void print();
};

// ========== CONSTRUCTORS ==========
template <typename T>
inline Rectangle<T>::Rectangle()
{}

template <typename T>
inline Rectangle<T>::Rectangle(T x, T y, T w, T h, bool valid)
    : mRoot(x,y,0,true), mSize(w,h,0,true), mValid(valid)
{}

template <typename T>
inline Rectangle<T>::Rectangle(T x, T y, Point<T> s, bool valid)
    : mRoot(x,y,0,true), mSize(s), mValid(valid)
{}

template <typename T>
inline Rectangle<T>::Rectangle(Point<T> r, T w, T h, bool valid)
    : mRoot(r), mSize(w,h,0,true), mValid(valid)
{}

template <typename T>
inline Rectangle<T>::Rectangle(Point<T> r, Point<T> s, bool valid)
    : mRoot(r), mSize(s), mValid(valid)
{}

// ========== UTILITY ==========

template <typename T>
void Rectangle<T>::print()
{
    printf("Rectangle type=%s, X=%s; Y=%s; W=%s; H=%s; Valid=%s\n",
        typeid(T).name(),
        std::to_string(mRoot.mX).c_str(),
        std::to_string(mRoot.mY).c_str(),
        std::to_string(mSize.mX).c_str(),
        std::to_string(mSize.mY).c_str(),
        mValid ? "true" : "false");
}

#endif // RECTANGLE_HPP