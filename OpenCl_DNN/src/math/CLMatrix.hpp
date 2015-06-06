/*
 * CLMatrix.h
 *
 *  Created on: Jun 2, 2015
 *      Author: hedi7
 */

#ifndef SRC_MATH_CLMATRIX_HPP_
#define SRC_MATH_CLMATRIX_HPP_


#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <cassert>
#include "OpenCL.hpp"


template <typename T>
class CL_Matrix {
public:
	CL_Matrix(u_int32_t r, u_int32_t c);
	CL_Matrix(CL_Matrix && other) noexcept;
	CL_Matrix(const CL_Matrix<T> &other);

	virtual ~CL_Matrix();

	void zeros();

	void random(T min,T max);

	void fill(T fill);

	CL_Matrix<T> transpose() const;

	CL_Matrix<T> dot(const CL_Matrix<T>& other) const;

// Forbid accesses with only one variable.
	T & operator[](u_int32_t n);
	T operator[](u_int32_t n) const;

	// Access the matrix coefficient at (r, c)
	T & operator()(u_int32_t r, u_int32_t c);

	// Access the matrix coefficient at (r, c)
	T operator()(u_int32_t r, u_int32_t c) const;

//	Componenet wise declarations
	CL_Matrix<T>& operator+=(CL_Matrix<T> const & mat);
	CL_Matrix<T>& operator-=(CL_Matrix<T> const & mat);
	CL_Matrix<T>& operator*=(CL_Matrix<T> const & mat);
// Multiply by constant
	CL_Matrix<T>& operator*=(T var);


	CL_Matrix<T>& operator=(const CL_Matrix  &mat);
	CL_Matrix<T>& operator=(CL_Matrix  mat);


	template<typename V>
	friend bool checkalign(CL_Matrix<V>const &lhs,CL_Matrix<V> const &rhs);

	template<typename V>
	friend bool checkdot(CL_Matrix<V>const &lhs,CL_Matrix<V> const &rhs);

	friend void swap<>(CL_Matrix<T> & lhs, CL_Matrix<T> & rhs);

private:



//	The acutal matrix behind it, we use std vector because the OpenCL
//	Interface is using it too, so we dont want to copy arrays around.
	std::vector<T> mat;
//	Number of columns
	u_int32_t _n_cols;

//	Number of rows
	u_int32_t _n_rows;

	const OpenCL _cl;

};


template <typename T>
CL_Matrix<T> operator+(CL_Matrix<T> lhs, CL_Matrix<T> const & rhs);


#include "CLMatrix.cpp"

#endif /* SRC_MATH_CLMATRIX_HPP_ */
