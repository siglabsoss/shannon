/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_COORDINATE_TRANSFORM__
#define __POP_COORDINATE_TRANSFORM__

#include <vector>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/tuple/tuple.hpp>

namespace pop
{

// An instance of this class defines a linear transformation that is determined
// by the (x, y, z) coordinates of three base stations. The transformation has
// the following properties:
//
//   1) Base station 1 is sent to (0, 0, 0).
//   2) Base station 2 is sent to a point on the X-axis. (I.e., its Y- and
//      Z-coordinates are 0).
//   3) Base station 3 is sent to a point on the X-Y plane. (I.e., its
//      Z-coordinate is 0).
//   4) Distances are preserved.
//
class PopCoordinateTransform
{
public:
	explicit PopCoordinateTransform(
		const std::vector<boost::tuple<double, double, double> >&
		base_stations);

	// Performs the transformation.
	boost::tuple<double, double, double> transform(
		const boost::tuple<double, double, double>& location) const;

	// Performs the inverse of the transformation. In other words, this method
	// converts a point in the transformed coordinate space into a point in the
	// original coordinate space.
	boost::tuple<double, double, double> untransform(
		const boost::tuple<double, double, double>& location) const;

private:
	typedef boost::numeric::ublas::matrix<double> Matrix;
	// TODO(snyderek): Rename this type to better distinguish it from
	// std::vector.
	typedef boost::numeric::ublas::vector<double> Vector;

	static boost::tuple<double, double> custom_atan2(double y, double x);

	static Matrix get_x_axis_rotation_matrix(double cos_theta,
											 double sin_theta);
	static Matrix get_y_axis_rotation_matrix(double cos_theta,
											 double sin_theta);
	static Matrix get_z_axis_rotation_matrix(double cos_theta,
											 double sin_theta);

	static Vector vector_from_tuple(
		const boost::tuple<double, double, double>& tup);
	static boost::tuple<double, double, double> tuple_from_vector(
		const Vector& v);

	Vector translation_vector_;
	Matrix rotation_matrix_;
	Matrix reverse_rotation_matrix_;
};

}

#endif
