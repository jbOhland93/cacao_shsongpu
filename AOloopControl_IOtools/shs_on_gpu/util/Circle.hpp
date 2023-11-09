#ifndef CIRCLE_HPP
#define CIRCLE_HPP

#include "Point.hpp"
#include "Pupil.hpp"

#define spCircle std::shared_ptr<Circle>

class Circle {
public:
    static spCircle makeCircle(Point<double> center, double radius);
    static spCircle makeCircle(const spPupil pupil);

    Point<double> toPolar(Point<double> cartesianPoint);
    Point<double> toCartesian(Point<double> polarPoint);
    bool insideCircle(Point<double> cartesianPoint);

    void print();

private:
    Point<double> mCenter;
    double mRadius;

    Circle(); // No publically available ctor
    Circle(Point<double> center, double radius) : mCenter(center), mRadius(radius) {}
};

#endif // CIRCLE_HPP
