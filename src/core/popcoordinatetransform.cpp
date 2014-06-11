/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <assert.h>
#include <math.h>

#include <vector>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/popcoordinatetransform.hpp"

using boost::get;
using boost::make_tuple;
using boost::numeric::ublas::prod;
using boost::tie;
using boost::tuple;
using std::vector;

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
	const vector<tuple<double, double, double> >& base_stations)
{
	assert(base_stations.size() >= 3u);

	// Create a vector for each of the base stations. This makes it easier to
	// apply linear transformations using Boost's linear algebra library.
	vector<Vector> bs_vectors(3);
	for (vector<Vector>::size_type i = 0; i < bs_vectors.size(); ++i) {
		bs_vectors[i] = vector_from_tuple(base_stations[i]);
	}

	// Compute the translation vector.
	translation_vector_ = -bs_vectors[0];

	// Apply the translation vector to each of the base stations.
	for (vector<Vector>::iterator it = bs_vectors.begin();
		 it != bs_vectors.end(); ++it) {
		*it += translation_vector_;
	}

	// Compute the Z-axis rotation.
	double cos_theta, sin_theta;
	tie(cos_theta, sin_theta) = custom_atan2(bs_vectors[1](1),
											 -bs_vectors[1](0));

	const Matrix rotation_matrix1 = get_z_axis_rotation_matrix(
		cos_theta, sin_theta);
	const Matrix reverse_rotation_matrix1 = get_z_axis_rotation_matrix(
		cos_theta, -sin_theta);

	// Apply the Z-axis rotation to the base stations.
	for (vector<Vector>::iterator it = bs_vectors.begin();
		 it != bs_vectors.end(); ++it) {
		*it = prod(rotation_matrix1, *it);
	}

	// Compute the Y-axis rotation.
	tie(cos_theta, sin_theta) = custom_atan2(bs_vectors[1](2),
											 bs_vectors[1](0));

	const Matrix rotation_matrix2 = get_y_axis_rotation_matrix(
		cos_theta, sin_theta);
	const Matrix reverse_rotation_matrix2 = get_y_axis_rotation_matrix(
		cos_theta, -sin_theta);

	// Apply the Y-axis rotation to the base stations.
	for (vector<Vector>::iterator it = bs_vectors.begin();
		 it != bs_vectors.end(); ++it) {
		*it = prod(rotation_matrix2, *it);
	}

	// Compute the X-axis rotation.
	tie(cos_theta, sin_theta) = custom_atan2(bs_vectors[2](2),
											 -bs_vectors[2](1));

	const Matrix rotation_matrix3 = get_x_axis_rotation_matrix(
		cos_theta, sin_theta);
	const Matrix reverse_rotation_matrix3 = get_x_axis_rotation_matrix(
		cos_theta, -sin_theta);

	// Apply the X-axis rotation to the base stations.
	for (vector<Vector>::iterator it = bs_vectors.begin();
		 it != bs_vectors.end(); ++it) {
		*it = prod(rotation_matrix3, *it);
	}

	// Combine the rotation matrices into a single matrix that will perform all
	// of the necessary rotations.
	Matrix temp = prod(rotation_matrix2, rotation_matrix1);
	rotation_matrix_ = prod(rotation_matrix3, temp);

	temp = prod(reverse_rotation_matrix1, reverse_rotation_matrix2);
	reverse_rotation_matrix_ = prod(temp, reverse_rotation_matrix3);
}

tuple<double, double, double> PopCoordinateTransform::transform(
	const tuple<double, double, double>& location) const
{
	// Translate and then rotate.
	return tuple_from_vector(
		prod(rotation_matrix_,
			 vector_from_tuple(location) + translation_vector_));
}

tuple<double, double, double> PopCoordinateTransform::untransform(
	const tuple<double, double, double>& location) const
{
	// Un-rotate and then un-translate.
	return tuple_from_vector(
		prod(reverse_rotation_matrix_, vector_from_tuple(location))
		- translation_vector_);
}

// This method is like atan2 in the standard C library, except that instead of
// returning an angle theta, it returns (cos(theta), sin(theta)). This is
// slightly more efficient than computing the angle and then calling cos() and
// sin() on it.

// static
tuple<double, double> PopCoordinateTransform::custom_atan2(double y, double x)
{
	const double dist = sqrt(x*x + y*y);
	return make_tuple(x / dist, y / dist);
}

// Each of the following methods returns a matrix for rotating a vector around
// the X-, Y-, or Z-axis.

// static
PopCoordinateTransform::Matrix
PopCoordinateTransform::get_x_axis_rotation_matrix(double cos_theta,
												   double sin_theta)
{
	Matrix m(3, 3);

	m(0, 0) = 1.0;
	m(0, 1) = 0.0;
	m(0, 2) = 0.0;
	m(1, 0) = 0.0;
	m(1, 1) = cos_theta;
	m(1, 2) = -sin_theta;
	m(2, 0) = 0.0;
	m(2, 1) = sin_theta;
	m(2, 2) = cos_theta;

	return m;
}

// static
PopCoordinateTransform::Matrix
PopCoordinateTransform::get_y_axis_rotation_matrix(double cos_theta,
												   double sin_theta)
{
	Matrix m(3, 3);

	m(0, 0) = cos_theta;
	m(0, 1) = 0.0;
	m(0, 2) = sin_theta;
	m(1, 0) = 0.0;
	m(1, 1) = 1.0;
	m(1, 2) = 0.0;
	m(2, 0) = -sin_theta;
	m(2, 1) = 0.0;
	m(2, 2) = cos_theta;

	return m;
}

// static
PopCoordinateTransform::Matrix
PopCoordinateTransform::get_z_axis_rotation_matrix(double cos_theta,
												   double sin_theta)
{
	Matrix m(3, 3);

	m(0, 0) = cos_theta;
	m(0, 1) = -sin_theta;
	m(0, 2) = 0.0;
	m(1, 0) = sin_theta;
	m(1, 1) = cos_theta;
	m(1, 2) = 0.0;
	m(2, 0) = 0.0;
	m(2, 1) = 0.0;
	m(2, 2) = 1.0;

	return m;
}

// static
PopCoordinateTransform::Vector PopCoordinateTransform::vector_from_tuple(
	const tuple<double, double, double>& tup)
{
	Vector v(3);
	v(0) = get<0>(tup);
	v(1) = get<1>(tup);
	v(2) = get<2>(tup);

	return v;
}

// static
tuple<double, double, double> PopCoordinateTransform::tuple_from_vector(
	const Vector& v)
{
	return make_tuple(v(0), v(1), v(2));
}

}
