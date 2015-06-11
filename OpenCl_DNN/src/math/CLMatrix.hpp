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
//	Inits with value T
	CL_Matrix(u_int32_t r, u_int32_t c,T value);
//	If true is given, it inits randomly between 0,1
	CL_Matrix(u_int32_t r, u_int32_t c,bool random);
	CL_Matrix(CL_Matrix && other) noexcept;
	CL_Matrix(const CL_Matrix<T> &other);

	virtual ~CL_Matrix();

	void zeros();

	void random(T min,T max);

	void fill(T fill);


	CL_Matrix<T> transpose() const;

//	Dot product between a matrix and another matrix
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

	CL_Matrix<T> operator*(CL_Matrix<T> const &other);

	void printDimension()const;

	CL_Matrix<T>& operator=(CL_Matrix  mat);

	template<typename V>
	friend CL_Matrix<V> operator-(CL_Matrix<V> const &lhs, CL_Matrix<V> const & rhs);

	template<typename V>
	friend CL_Matrix<V> operator+(CL_Matrix<V> const &lhs, CL_Matrix<V> const & rhs);

	template<typename V>
	friend CL_Matrix<V> operator*(V val, CL_Matrix<V> const & rhs);

	template<typename V>
	friend void checkalign(CL_Matrix<V>const &lhs,CL_Matrix<V> const &rhs);

	template<typename V>
	friend void checkdot(CL_Matrix<V>const &lhs,CL_Matrix<V> const &rhs);

	friend inline void swap(CL_Matrix & lhs, CL_Matrix& rhs){
		using std::swap;
		swap(lhs.mat,rhs.mat);
		swap(lhs._n_rows,rhs._n_rows);
		swap(lhs._n_cols,rhs._n_cols);
		swap(lhs._cl,rhs._cl);
	}

	std::pair<int,int> getDimensions();

//	Computes sigmoid function from this object and returns the result
	CL_Matrix<T> sigmoid() const;

	CL_Matrix<T> sigmoidgrad() const;

//	Computes tanh function and returns the result
	CL_Matrix<T> tanh() const;

	u_int32_t getCols() const {
		return _n_cols;
	}

	u_int32_t getRows() const {
		return _n_rows;
	}

	template< typename V>
	friend std::ostream &operator<<(std::ostream &output, const CL_Matrix<V> &mat);

private:



//	The acutal matrix behind it, we use std vector because the OpenCL
//	Interface is using it too, so we dont want to copy arrays around.
	std::vector<T> mat;
//	Number of columns
	u_int32_t _n_cols;

//	Number of rows
	u_int32_t _n_rows;

	OpenCL _cl;

};

//tempalte
//ostream &operator<<(ostream &output, const CL_Matrix<T> &mat){
//
//}




#include "CLMatrix.cpp"

#endif /* SRC_MATH_CLMATRIX_HPP_ */
