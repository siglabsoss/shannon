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

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/tuple/tuple.hpp>

namespace pop
{

// An instance of this class defines a linear transformation that is determined
// by the (x, y, z) coordinates of two base stations. The transformation has the
// following properties:
//
//   1) Base station 1 is moved to (0, 0, 0).
//   2) Base station 2 is moved to a point on the X-axis.
//   3) Distances are preserved.
//
class PopCoordinateTransform
{
public:
	PopCoordinateTransform(
		const boost::tuple<double, double, double>& base_station1,
		const boost::tuple<double, double, double>& base_station2);

	// Performs the transformation.
	boost::tuple<double, double, double> transform(
		const boost::tuple<double, double, double>& location) const;
	// Performs the inverse of the transformation.
	boost::tuple<double, double, double> untransform(
		const boost::tuple<double, double, double>& location) const;

private:
	typedef boost::numeric::ublas::matrix<double> Matrix;
	typedef boost::numeric::ublas::vector<double> Vector;

	static Matrix get_rotation_matrix(const Vector& axis, double cos_theta,
									  double sin_theta);
	static Vector unit_vector(const Vector& v);
	static Vector vector_from_coords(double x, double y, double z);
	static Vector vector_from_tuple(
		const boost::tuple<double, double, double>& tup);
	static boost::tuple<double, double, double> tuple_from_vector(
		const Vector& v);

	const Vector translation_vector_;
	Matrix rotation_matrix_;
	Matrix reverse_rotation_matrix_;
};

}

#endif
