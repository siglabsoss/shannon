/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <math.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/popcoordinatetransform.hpp"

using boost::get;
using boost::make_tuple;
using boost::numeric::ublas::prod;
using boost::tuple;

namespace pop
{

namespace
{

double sqr(double x)
{
	return x * x;
}

}  // namespace

PopCoordinateTransform::PopCoordinateTransform(
	const tuple<double, double, double>& base_station1,
	const tuple<double, double, double>& base_station2)
	: translation_vector_(-vector_from_tuple(base_station1))
{
	// The baseline vector is a unit vector that indicates the direction from
	// base station 1 to base station 2.
	const Vector baseline =
		unit_vector(vector_from_tuple(base_station2) + translation_vector_);

	// Let (x, y, z) be the coordinates of the baseline vector. We want to
	// rotate this vector around the origin so that it's aligned with the
	// X-axis. The axis of rotation must be orthogonal to both (x, y, z) and
	// (1, 0, 0). To find this vector, take the cross product of (x, y, z) and
	// (1, 0, 0). The cross product is (0, z, -y).
	//
	// Let theta be the angle (in radians) through which the baseline vector
	// must be rotated to make it aligned with the X-axis. Since (x, y, z) and
	// (1, 0, 0) are both unit vectors, cos(theta) is equal to the dot product
	// of (x, y, z) and (1, 0, 0). Therefore, cos(theta) = x.

	const double x = baseline(0);
	const double y = baseline(1);
	const double z = baseline(2);

	const Vector axis_of_rotation =
		unit_vector(vector_from_coords(0.0, z, -y));
	const double cos_theta = x;
	// By the Pythagorean theorem, sqr(cos_theta) + sqr(sin_theta) = 1.
	const double sin_theta = sqrt(1.0 - sqr(cos_theta));

	rotation_matrix_ = get_rotation_matrix(axis_of_rotation, cos_theta,
										   sin_theta);
	// Trigonometric identities:
	//   cos(-theta) == cos(theta)
	//   sin(-theta) == -sin(theta)
	reverse_rotation_matrix_ = get_rotation_matrix(axis_of_rotation, cos_theta,
												   -sin_theta);
}

tuple<double, double, double> PopCoordinateTransform::transform(
	const tuple<double, double, double>& location) const
{
	return tuple_from_vector(
		prod(rotation_matrix_,
			 vector_from_tuple(location) + translation_vector_));
}

tuple<double, double, double> PopCoordinateTransform::untransform(
	const tuple<double, double, double>& location) const
{
	return tuple_from_vector(
		prod(reverse_rotation_matrix_, vector_from_tuple(location))
		- translation_vector_);
}

// Returns the matrix to rotate a vector by 'theta' radians around the vector
// 'axis'. 'axis' must be a unit vector.
//
// http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/
// Section 5.2
// Accessed June 9, 2014.

// static
PopCoordinateTransform::Matrix PopCoordinateTransform::get_rotation_matrix(
	const Vector& axis, double cos_theta, double sin_theta)
{
	const double u = axis(0);
	const double v = axis(1);
	const double w = axis(2);

	const double one_minus_cos = 1.0 - cos_theta;

	Matrix m(3, 3);

	m(0, 0) = u*u + (1.0 - u*u) * cos_theta;
	m(0, 1) = u*v * one_minus_cos - w * sin_theta;
	m(0, 2) = u*w * one_minus_cos + v * sin_theta;
	m(1, 0) = u*v * one_minus_cos + w * sin_theta;
	m(1, 1) = v*v + (1.0 - v*v) * cos_theta;
	m(1, 2) = v*w * one_minus_cos - u * sin_theta;
	m(2, 0) = u*w * one_minus_cos - v * sin_theta;
	m(2, 1) = v*w * one_minus_cos + u * sin_theta;
	m(2, 2) = w*w + (1.0 - w*w) * cos_theta;

	return m;
}

// static
PopCoordinateTransform::Vector PopCoordinateTransform::unit_vector(
	const Vector& v)
{
	const double magnitude = sqrt(sqr(v(0)) + sqr(v(1)) + sqr(v(2)));
	return v / magnitude;
}

// static
PopCoordinateTransform::Vector PopCoordinateTransform::vector_from_coords(
	double x, double y, double z)
{
	Vector v(3);
	v(0) = x;
	v(1) = y;
	v(2) = z;

	return v;
}

// static
PopCoordinateTransform::Vector PopCoordinateTransform::vector_from_tuple(
	const tuple<double, double, double>& tup)
{
	return vector_from_coords(get<0>(tup), get<1>(tup), get<2>(tup));
}

// static
tuple<double, double, double> PopCoordinateTransform::tuple_from_vector(
	const Vector& v)
{
	return make_tuple(v(0), v(1), v(2));
}

}
