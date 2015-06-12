/*
 * CLMatrix.cpp
 *
 *  Created on: Jun 2, 2015
 *      Author: hedi7
 */


template<typename T>
inline CL_Matrix<T>::CL_Matrix(u_int32_t r, u_int32_t c):
_n_rows(r),_n_cols(c),mat(r*c),_cl(OpenCLPort::getInstance("kernels.cl")){
}

template<typename T>
inline CL_Matrix<T>::CL_Matrix(u_int32_t r, u_int32_t c, T value):
	CL_Matrix<T>(r,c){
	this->fill(value);
}

template<typename T>
inline CL_Matrix<T>::CL_Matrix(u_int32_t r, u_int32_t c, bool random):
	CL_Matrix<T>(r,c){
	if (random){
		this->random(0,1);
	}
}


template<typename T>
inline CL_Matrix<T>::~CL_Matrix() {
}

template<typename T>
inline CL_Matrix<T>::CL_Matrix(const CL_Matrix<T> &other):_cl(other._cl),_n_cols(other._n_cols),_n_rows(other._n_rows),mat(other.mat){
}

template <typename T>
CL_Matrix<T>::CL_Matrix(CL_Matrix && other) noexcept :CL_Matrix(other._n_rows,other._n_cols)
{
	swap(*this,other);
}


template<typename T>
CL_Matrix<T>& CL_Matrix<T>::operator=(CL_Matrix<T> other){
	swap(*this,other);
	return (*this);
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
	std::generate(this->mat.begin(), this->mat.end(),[&]{ return dist(rd); });
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

inline int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

template<typename T>
CL_Matrix<T> CL_Matrix<T>::dot(const CL_Matrix<T>& other) const{
	checkdot(*this,other);
//	initzialize the result matrix
	CL_Matrix res(this->_n_rows, other._n_cols);
//	Last argument is the output argument

	std::size_t localrows = ceil(float(this->_n_rows)/100);
	std::size_t localcols = ceil(float(this->_n_cols)/100);
	std::vector<std::size_t> outputargs = {4};
//	Currently unused, crashes unfortunately even if hardcoded args are given at a certain size
	std::vector<std::size_t> localWorkSize = {localrows,localcols};
	std::vector<std::size_t> globalWorkSize = {this->_n_rows,other._n_cols};

    this->_cl.runKernel("mat_mul",outputargs,globalWorkSize,localWorkSize,this->mat,other.mat,this->_n_cols,other._n_cols,res.mat);
	return res;
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
	assert(r*c < this->mat.size());
	return this->mat.at(r * this->_n_cols + c);
}

template<typename T>
inline T CL_Matrix<T>::operator ()(u_int32_t r, u_int32_t c) const {
	assert(r*c < this->mat.size());
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
inline CL_Matrix<T> CL_Matrix<T>::sigmoid() const{
	CL_Matrix res(this->_n_rows, this->_n_cols);
	std::vector<std::size_t> outputargs = {2};
	std::vector<std::size_t> localWorkSize = {1,1};
	std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
	this->_cl.runKernel("sigmoid",outputargs,globalWorkSize,localWorkSize,res._n_cols,this->mat,res.mat);
	return res;
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::sigmoidgrad() const {
	CL_Matrix res(this->_n_rows, this->_n_cols);
	std::vector<std::size_t> outputargs = {2};
	std::vector<std::size_t> localWorkSize = {1,1};
	std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
	this->_cl.runKernel("sigmoidgrad",outputargs,globalWorkSize,localWorkSize,res._n_cols,this->mat,res.mat);
	return res;
}

template<typename T>
inline void CL_Matrix<T>::printDimension()const {
	std::cout << "Rows : " << this->_n_rows << " Cols : " << this->_n_cols << std::endl;
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::operator *(const CL_Matrix<T>& other) {
//	CHeck if size is the same
	checkalign(*this,other);
	CL_Matrix<T> res(other._n_rows,other._n_cols);
	for(unsigned i = 0; i < res.mat.size(); ++i) {
		res.mat.at(i) = this->mat.at(i) * other.mat.at(i);
	}
	return res;
}

template<typename T>
inline std::pair<int, int> CL_Matrix<T>::getDimensions() {
	return std::make_pair(this->_n_rows,this->_n_cols);
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::subMatCol(u_int32_t c) {
	assert(c<this->_n_cols);
	std::vector<T> col;
	for(auto i = c; i < this->mat.size(); i+=_n_cols){
		col.push_back(this->mat.at(i));
	}
	CL_Matrix<T> res(this->_n_rows,1,col);
	return res;
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::subMatRow(u_int32_t r) {
	assert(r < this->_n_rows);
	typename std::vector<T>::const_iterator first = this->mat.begin()+r*this->_n_cols;
	typename std::vector<T>::const_iterator last = this->mat.begin()+(r+1)*this->_n_cols;
	std::vector<T> row(first,last);
	CL_Matrix<T> res(1,this->_n_cols,row);
	return res;

}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::tanh() const{
	CL_Matrix<T> res(this->_n_rows, this->_n_cols);
	std::vector<std::size_t> outputargs = {2};
	std::vector<std::size_t> localWorkSize = {1,1};
	std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
	this->_cl.runKernel("cl_tanh",outputargs,globalWorkSize,localWorkSize,res._n_cols,this->mat,res.mat);
	return res;
}

template<typename T>
inline CL_Matrix<T> operator +(CL_Matrix<T> const &lhs,  CL_Matrix<T> const &rhs) {
	checkalign(lhs,rhs);
	CL_Matrix<T> res(lhs._n_rows,rhs._n_cols);
	for(unsigned i = 0; i < res.mat.size(); ++i) {
		res.mat.at(i) = lhs.mat.at(i) + rhs.mat.at(i);
	}
	return res;
}
template<typename T>
inline CL_Matrix<T> operator -(CL_Matrix<T> const &lhs,  CL_Matrix<T> const &rhs) {
	checkalign(lhs,rhs);
	CL_Matrix<T> res(lhs._n_rows,rhs._n_cols);
	for(unsigned i = 0; i < res.mat.size(); ++i) {
		res.mat.at(i) = lhs.mat.at(i) - rhs.mat.at(i);
	}
	return res;

}

template<typename T>
inline CL_Matrix<T> operator*(T val, CL_Matrix<T> const &rhs) {
	CL_Matrix<T> res(rhs._n_rows,rhs._n_cols);
	for(unsigned i = 0; i < res.mat.size(); ++i) {
		res.mat.at(i) = val*rhs.mat.at(i);
	}
	return res;
}

template<typename T>
std::ostream &operator<<(std::ostream &output, const CL_Matrix<T> &mat){
	for(auto i = 0u; i <mat._n_rows;i++ ){
		for(auto j = 0u; j <mat._n_cols;j++ ){
			output << mat(i,j) << " ";
		}
		output << "\n";
	}
	return output;
}

template<typename T>
void checkalign(const CL_Matrix<T>& lhs, const CL_Matrix<T>& rhs) {
	assert(lhs._n_cols == rhs._n_cols);
	assert(lhs._n_rows == rhs._n_rows);
}

template<typename T>
void checkdot(const CL_Matrix<T>& lhs, const CL_Matrix<T>& rhs) {
	assert(lhs._n_cols == rhs._n_rows);
}

