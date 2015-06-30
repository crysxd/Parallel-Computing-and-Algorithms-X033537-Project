/*
 * CLMatrix.cpp
 *
 *  Created on: Jun 2, 2015
 *      Author: hedi7
 */


template<typename T>
inline CL_Matrix<T>::CL_Matrix(u_int32_t r, u_int32_t c):
state(OnlyRam),_n_rows(r),_n_cols(c),mat(r*c),_cl(OpenCLPort::getInstance("kernels.cl")){
	this->fill(0);
}

template<typename T>
inline CL_Matrix<T>::CL_Matrix(u_int32_t r, u_int32_t c, T value):
state(OnlyRam),_n_rows(r),_n_cols(c),mat(r*c),_cl(OpenCLPort::getInstance("kernels.cl")){
	this->fill(value);

}

template<typename T>
inline CL_Matrix<T>::CL_Matrix(u_int32_t r, u_int32_t c, bool random):
	CL_Matrix<T>(r,c){
	if (random){
        this->random(-0.5,0.5);
	}
}


template<typename T>
inline CL_Matrix<T>::~CL_Matrix() {
}

template<typename T>
inline CL_Matrix<T>::CL_Matrix(const CL_Matrix<T> &other):_cl(other._cl),_n_cols(other._n_cols),_n_rows(other._n_rows),mat(other.mat),gpu_buf(other.gpu_buf),state(other.state){
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
    this->moveToRam();
	this->fill(0);
}

template<typename T>
inline void CL_Matrix<T>::random(T min,T max) {
    this->moveToRam();
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<T> dist(min,max);
	std::generate(this->mat.begin(), this->mat.end(),[&]{ return dist(rd); });
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::transpose() const {
    this->syncToRam();
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
CL_Matrix<T> CL_Matrix<T>::dot(const CL_Matrix<T>& other) const {
	checkdot(*this,other);
//	initzialize the result matrix
	CL_Matrix res(this->_n_rows, other._n_cols);
    this->dot(other, &res);
	return res;
}

template<typename T>
void CL_Matrix<T>::dot(const CL_Matrix<T>& other, CL_Matrix<T> *out) const {
    checkdot(*this,other);
    assert(out->_n_rows == this->_n_rows);
    assert(out->_n_cols == other._n_cols);

    out->moveToGpu();
    this->syncToGpu();
    other.syncToGpu();

    std::size_t localrows = ceil(float(this->_n_rows)/100);
    std::size_t localcols = ceil(float(this->_n_cols)/100);
//	Currently unused, crashes unfortunately even if hardcoded args are given at a certain size
    std::vector<std::size_t> localWorkSize = {localrows,localcols};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,other._n_cols};

    this->_cl.runKernelnoOut("mat_mul",globalWorkSize,localWorkSize,this->gpu_buf,other.gpu_buf,this->_n_cols,other._n_cols,out->gpu_buf);
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
    this->syncToRam();
    assert(r*c < this->mat.size());
    return this->mat.at(r * this->_n_cols + c);
}

template<typename T>
inline T CL_Matrix<T>::operator ()(u_int32_t r, u_int32_t c) const {
    this->syncToRam();
    assert(r*c < this->mat.size());
    return this->mat.at( r * this->_n_cols + c);
}

template<typename T>
inline CL_Matrix<T>& CL_Matrix<T>::operator +=(const CL_Matrix<T>& other) {
	checkalign(*this,other);
//    for(unsigned i = 0; i < this->mat.size(); ++i) {
//        this->mat.at(i) += other.mat.at(i);
//    }
    this->moveToGpu();
    other.syncToGpu();
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("add",globalWorkSize,localWorkSize,other._n_cols,this->gpu_buf,other.gpu_buf,this->gpu_buf);
    return (*this);
}

template<typename T>
inline CL_Matrix<T>& CL_Matrix<T>::operator -=(const CL_Matrix<T>& other) {
    checkalign(*this,other);
//    for(unsigned i = 0; i < this->mat.size(); ++i) {
//        this->mat.at(i) -= other.mat.at(i);
//    }
    this->moveToGpu();
    other.syncToGpu();
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("sub",globalWorkSize,localWorkSize,other._n_cols,this->gpu_buf,other.gpu_buf,this->gpu_buf);
    return (*this);
}

template<typename T>
inline CL_Matrix<T>& CL_Matrix<T>::operator *=(const CL_Matrix<T>& other) {
    checkalign(*this,other);
//    for(unsigned i = 0; i < this->mat.size(); ++i) {
//        this->mat.at(i) *= other.mat.at(i);
//    }

    this->moveToGpu();
    other.syncToGpu();
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("mul",globalWorkSize,localWorkSize,other._n_cols,this->gpu_buf,other.gpu_buf,this->gpu_buf);
    return (*this);
}

template<typename T>
inline void CL_Matrix<T>::fill(T fill) {
    this->moveToRam();
	std::fill(this->mat.begin(), this->mat.end(), fill);
}


template<typename T>
inline void CL_Matrix<T>::syncToGpu() const {
    if (this->state == OnlyRam) {
//        std::cout << "transfer " << _n_rows << 'x' << _n_cols << " to gpu\n";
        this->gpu_buf = this->_cl.putDataOnGPU(this->mat);
        this->state = Synced;
    }

}

template<typename T>
inline void CL_Matrix<T>::syncToRam() const {
    if (this->state == OnlyGpu) {
//        std::cout << "transfer " << _n_rows << 'x' << _n_cols << " to ram\n";
        this->_cl.readBuffer(this->gpu_buf,this->mat);
        this->state = Synced;
    }
}

template<typename T>
inline void CL_Matrix<T>::moveToGpu() {
    this->syncToGpu();
    this->state = OnlyGpu;

}

template<typename T>
inline void CL_Matrix<T>::moveToRam() {
    this->syncToRam();
    this->state = OnlyRam;
}



/**
 *
 */
template<typename T>
inline void CL_Matrix<T>::shuffle(bool row){
	// Shuffles the row or column of the matrix depending on the
	// given boolean value
    this->moveToRam();
	std::random_device rd;
	std::mt19937 mt(rd());
	std::vector<std::pair<int,int>> indices;
	if (row){
		for (auto i = 0u; i < this->mat.size(); i+=_n_cols)
		{
			indices.push_back(std::make_pair(i,i+_n_cols-1));
		}
		std::shuffle(indices.begin(),indices.end(),mt);
		auto matind=0u;
		for(std::pair<int,int> &e :indices){
			for(auto j = e.first; j <= e.second ; j++){
				std::swap(this->mat[matind],this->mat[j]);
				matind ++;
			}
		}
	}
	else{
		for (auto i = 0u; i <_n_rows; i++)
		{
			indices.push_back(std::make_pair(i,i+_n_rows*(_n_cols-1)));
		}
		std::shuffle(indices.begin(),indices.end(),mt);
		auto matind=0u;
		for(std::pair<int,int> &e :indices){
			for(auto j = e.first; j <= e.second ; j+= _n_rows){
				std::swap(this->mat[matind],this->mat[j]);
				matind= (matind+_n_rows)%(this->mat.size());
			}
			matind++;
		}
	}
}

template<typename T>
inline void CL_Matrix<T>::fillAt(u_int32_t r, u_int32_t c, T value) {
    this->moveToRam();
	assert(r*c < this->mat.size());
	this->mat.at( r * this->_n_cols + c) = value;
}


template<typename T>
inline CL_Matrix<T>& CL_Matrix<T>::operator *=(T var) {
//    for(unsigned i = 0; i < this->mat.size(); ++i) {
//        this->mat.at(i) *= var;
//    }
    this->moveToGpu();
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("mul_scalar",globalWorkSize,localWorkSize,this->_n_cols,this->gpu_buf,var,this->gpu_buf);
    return (*this);
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::sigmoid() const{
	CL_Matrix res(this->_n_rows, this->_n_cols);
    this->sigmoid(&res);
	return res;
}

template<typename T>
void CL_Matrix<T>::sigmoid(CL_Matrix<T> *out) const {
    this->syncToGpu();
    out->moveToGpu();
    assert(out->_n_rows == this->_n_rows);
    assert(out->_n_cols == this->_n_cols);
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("sigmoid",globalWorkSize,localWorkSize,out->_n_cols,this->gpu_buf,out->gpu_buf);
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::sigmoidcpu() const {

	CL_Matrix res(this->_n_rows,this->_n_cols);
	for(auto i=0u; i < this->mat.size();i++){
		res.mat.at(i) = 1.f/(1.f+exp(this->mat.at(i)));
	}
	return res;
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::sigmoidgradcpu() const {
	CL_Matrix res(this->_n_rows,this->_n_cols);
	for(auto i=0u; i < this->mat.size();i++){
		res.mat.at(i) = 1.f/(1.f+exp(this->mat.at(i)))* (1- (1.f/(1.f+exp(this->mat.at(i)))));
	}
	return res;
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::sigmoidgrad() const{
    CL_Matrix res(this->_n_rows, this->_n_cols);
    this->sigmoidgrad(&res);
    return res;
}

template<typename T>
void CL_Matrix<T>::sigmoidgrad(CL_Matrix<T> *out) const {
    this->syncToGpu();
    out->moveToGpu();
    assert(out->_n_rows == this->_n_rows);
    assert(out->_n_cols == this->_n_cols);
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("sigmoidgrad",globalWorkSize,localWorkSize,out->_n_cols,this->gpu_buf,out->gpu_buf);
}

template<typename T>
inline CL_Matrix<T>::operator T const &() const{
    this->syncToRam();
	if(_n_cols!=1 && _n_rows !=1){
		throw "size not 1x1";
	}
	return *(this)(0,0);
}

template<typename T>
inline CL_Matrix<T>::operator T &(){
    this->syncToRam();
	if(_n_cols!=1 && _n_rows !=1){
		throw "size not 1x1";
	}
    return (*this)( 0, 0 );

}

template<typename T>
inline void CL_Matrix<T>::printDimension()const {
	std::cout << "Rows : " << this->_n_rows << " Cols : " << this->_n_cols << std::endl;
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::operator *(const CL_Matrix<T>& other) {
	CL_Matrix<T> res(other._n_rows,other._n_cols);
    this->mul(other, &res);
	return res;
}

template<typename T>
void CL_Matrix<T>::mul(const CL_Matrix<T>& other, CL_Matrix<T> *out) const {
    checkalign(*this,other);
    checkalign(other,*out);
    this->syncToGpu();
    other.syncToGpu();
    out->moveToGpu();
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("mul",globalWorkSize,localWorkSize,out->_n_cols,this->gpu_buf,other.gpu_buf,out->gpu_buf);
}

template<typename T>
void CL_Matrix<T>::add(const CL_Matrix<T>& other, CL_Matrix<T> *out) const {
    checkalign(*this,other);
    checkalign(other,*out);
    this->syncToGpu();
    other.syncToGpu();
    out->moveToGpu();
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("add",globalWorkSize,localWorkSize,out->_n_cols,this->gpu_buf,other.gpu_buf,out->gpu_buf);
}

template<typename T>
void CL_Matrix<T>::sub(const CL_Matrix<T>& other, CL_Matrix<T> *out) const {
    checkalign(*this,other);
    checkalign(other,*out);
    this->syncToGpu();
    other.syncToGpu();
    out->moveToGpu();
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("sub",globalWorkSize,localWorkSize,out->_n_cols,this->gpu_buf,other.gpu_buf,out->gpu_buf);
}

template<typename T>
inline std::pair<int, int> CL_Matrix<T>::getDimensions() {
	return std::make_pair(this->_n_rows,this->_n_cols);
}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::subMatCol(u_int32_t c) {
    this->syncToRam();
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
    this->syncToRam();
	assert(r < this->_n_rows);
	typename std::vector<T>::const_iterator first = this->mat.begin()+r*this->_n_cols;
	typename std::vector<T>::const_iterator last = this->mat.begin()+(r+1)*this->_n_cols;
	std::vector<T> row(first,last);
	CL_Matrix<T> res(1,this->_n_cols,row);
	return res;

}

template<typename T>
inline CL_Matrix<T> CL_Matrix<T>::tanh() const{
    CL_Matrix res(this->_n_rows, this->_n_cols);
    this->tanh(&res);
    return res;
}

template<typename T>
void CL_Matrix<T>::tanh(CL_Matrix<T> *out) const {
    this->syncToGpu();
    out->moveToGpu();
    assert(out->_n_rows == this->_n_rows);
    assert(out->_n_cols == this->_n_cols);
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {this->_n_rows,this->_n_cols};
    this->_cl.runKernelnoOut("cl_tanh",globalWorkSize,localWorkSize,out->_n_cols,this->gpu_buf,out->gpu_buf);
}

template<typename T>
inline CL_Matrix<T> operator +(CL_Matrix<T> const &lhs,  CL_Matrix<T> const &rhs) {
    checkalign(lhs,rhs);
	CL_Matrix<T> res(lhs._n_rows,rhs._n_cols);
    lhs.add(rhs, &res);
//    for(unsigned i = 0; i < res.mat.size(); ++i) {
//		res.mat.at(i) = lhs.mat.at(i) + rhs.mat.at(i);
//	}
	return res;
}
template<typename T>
inline CL_Matrix<T> operator -(CL_Matrix<T> const &lhs,  CL_Matrix<T> const &rhs) {
    checkalign(lhs,rhs);
	CL_Matrix<T> res(lhs._n_rows,rhs._n_cols);
    lhs.sub(rhs, &res);
//	for(unsigned i = 0; i < res.mat.size(); ++i) {
//		res.mat.at(i) = lhs.mat.at(i) - rhs.mat.at(i);
//	}
	return res;

}

template<typename T>
inline CL_Matrix<T> operator*(T val, CL_Matrix<T> const &rhs) {
    rhs.syncToGpu();
    CL_Matrix<T> res(rhs._n_rows,rhs._n_cols);
    res.moveToGpu();
    std::vector<std::size_t> localWorkSize = {1,1};
    std::vector<std::size_t> globalWorkSize = {rhs._n_rows,rhs._n_cols};
    rhs._cl.runKernelnoOut("mul_scalar",globalWorkSize,localWorkSize,rhs._n_cols,rhs.gpu_buf,val,res.gpu_buf);
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

