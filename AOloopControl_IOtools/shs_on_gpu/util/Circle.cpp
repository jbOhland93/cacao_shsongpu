#include "Circle.hpp"
#include <cmath>

spCircle Circle::makeCircle(Point<double> center, double radius)
{
    return spCircle(new Circle(center, radius));
}

spCircle Circle::makeCircle(const spPupil pupil)
{
    int width = pupil->getWidth();
    int height = pupil->getHeight();
    int arrSize;
    bool* arr = pupil->getDataPtr(&arrSize);

    Point<double> center(0, 0);
    for (int ix = 0; ix < width; ix++)
        for (int iy = 0; iy < width; iy++)
            if (arr[iy*width + ix])
                center += Point<double>(ix, iy);
    center /= pupil->getNumValidFields();

    double radius = sqrt(pupil->getNumValidFields() / M_PI);
    
    return std::shared_ptr<Circle>(new Circle(center, radius));
}

Point<double> Circle::toPolar(Point<double> cartesianPoint)
{
    double dx = cartesianPoint.mX - mCenter.mX;
    double dy = cartesianPoint.mY - mCenter.mY;
    double r = std::sqrt(dx*dx + dy*dy) / mRadius;
    double theta = std::atan2(dy, dx);
    return Point<double>(r, theta);
}

Point<double> Circle::toCartesian(Point<double> polarPoint)
{
    double r = polarPoint.mX * mRadius;
    double theta = polarPoint.mY;
    double x = r * std::cos(theta) + mCenter.mX;
    double y = r * std::sin(theta) + mCenter.mY;
    return Point<double>(x, y);
}

bool Circle::insideCircle(Point<double> cartesianPoint)
{
    return toPolar(cartesianPoint).mX;
}

void Circle::print()
{
    printf("Circle center: ");
    mCenter.print();
    printc("Circle radius: %.3f\n", mRadius);
}
