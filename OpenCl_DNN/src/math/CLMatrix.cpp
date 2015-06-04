/*
 * CLMatrix.cpp
 *
 *  Created on: Jun 2, 2015
 *      Author: hedi7
 */


template<typename T>
inline CL_Matrix<T>::CL_Matrix(u_int32_t r, u_int32_t c):
	_n_cols(c),_n_rows(r),mat(r*c),_cl("matmul.cl"){

}

template<typename T>
inline CL_Matrix<T>::~CL_Matrix() {
}

template<typename T>
inline CL_Matrix<T>::CL_Matrix(CL_Matrix<T> &other){
//	this->_cl = other._cl;
	this->_n_cols = other._n_cols;
	this->_n_rows = other._n_rows;
	this->mat = other.mat;
}

template<typename T>
CL_Matrix<T>& CL_Matrix<T>::operator=(const CL_Matrix<T> &other){
	std::swap(*this,other);
	return *this;
}


template<typename T>
inline void CL_Matrix<T>::zeros() {
	std::fill(this->mat.begin(), this->mat.end(), 0);
}

template<typename T>
inline void CL_Matrix<T>::random(T min,T max) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<T> dist(min,max);
	std::generate(this->mat.begin(), this->mat.end(),[&dist,&rd]{ return dist(rd); });
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::transpose() const {
	CL_Matrix<T> trans(this->_n_cols,this->_n_rows);
    for(unsigned i = 0; i < this->_n_rows; ++i) {
        for(unsigned j = 0; j < this->_n_cols; ++j) {
            trans(j,i) = (*this)(i,j);
        }
    }
    return trans;
}

template<typename T>
CL_Matrix<T> CL_Matrix<T>::dot(const CL_Matrix<T>& other) const {
//	checkdot(*this,other);

//    this->_cl_intf.runKernel("matmul.cl",1,inp,out);
//	return (*this);

}

template<typename T>
inline T& CL_Matrix<T>::operator [](u_int32_t n) {
	throw "Not Implemented, please use operator (r,c)";
}

template<typename T>
inline T CL_Matrix<T>::operator [](u_int32_t n) const {
	throw "Not Implemented, please use operator (r,c)";
}

template<typename T>
inline T& CL_Matrix<T>::operator ()(u_int32_t r, u_int32_t c) {
	return this->mat.at(r * this->_n_cols + c);
}

template<typename T>
inline T CL_Matrix<T>::operator ()(u_int32_t r, u_int32_t c) const {
	return this->mat.at( r * this->_n_cols + c);
}

template<typename T>
inline CL_Matrix<T>& CL_Matrix<T>::operator +=(const CL_Matrix<T>& other) {
	checkalign(*this,other);
    for(unsigned i = 0; i < this->mat.size(); ++i) {
        this->mat.at(i) += other.mat.at(i);
    }
    return (*this);
}

template<typename T>
inline CL_Matrix<T>& CL_Matrix<T>::operator -=(const CL_Matrix<T>& other) {
	checkalign(*this,other);
    for(unsigned i = 0; i < this->mat.size(); ++i) {
        this->mat.at(i) -= other.mat.at(i);
    }
    return (*this);
}

template<typename T>
inline CL_Matrix<T>& CL_Matrix<T>::operator *=(const CL_Matrix<T>& other) {
	checkalign(*this,other);
    for(unsigned i = 0; i < this->mat.size(); ++i) {
        this->mat.at(i) *= other.mat.at(i);
    }
    return (*this);
}

template<typename T>
inline void CL_Matrix<T>::fill(T fill) {
	std::fill(this->mat.begin(), this->mat.end(), fill);
}

template<typename T>
inline CL_Matrix<T>& CL_Matrix<T>::operator *=(T var) {
    for(unsigned i = 0; i < this->mat.size(); ++i) {
        this->mat.at(i) *= var;
    }
    return (*this);
}

template<typename T>
inline CL_Matrix<T> operator +(CL_Matrix<T> lhs, const CL_Matrix<T>& rhs) {
}

template<typename T>
bool checkalign(const CL_Matrix<T>& lhs, const CL_Matrix<T>& rhs) {
	assert(lhs._n_cols == rhs._n_cols);
	assert(lhs._n_rows == rhs._n_rows);
}

template<typename T>
bool checkdot(const CL_Matrix<T>& lhs, const CL_Matrix<T>& rhs) {
	assert(lhs._n_cols == rhs._n_rows);
}

