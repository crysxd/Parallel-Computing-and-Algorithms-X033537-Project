#include "gtest/gtest.h"
#include "Activation.h"
#include "HiddenLayer.h"
#include "Sigmoid.h"
#include "TanH.h"

#include <vector>
#include "util.hpp"
#include "OpenCL.hpp"
#include "CLMatrix.hpp"
#include "FeedForwardNN.h"

using namespace std;

TEST(Matrix,init){
    int r = 25;
    int c = 30;
    CL_Matrix<float> mat(r,c);
    mat.zeros();
    for (int i = 0; i < r; ++i)
    {
        for(unsigned j = 0; j < c; ++j) {
            EXPECT_EQ(mat(i,j),0);
        }
    }
    // Randomize matrix
    mat.random(0.5,1);

    for (int i = 0; i < r; ++i)
    {
        for(unsigned j = 0; j < c; ++j) {
            EXPECT_NEAR(mat(i,j),0.75,0.25);
        }
    }

}

TEST(Matrix,transpose){
    int r = 25;
    int c = 30;
    CL_Matrix<float> mat(r,c);
    mat.random(0,1);
    CL_Matrix<float> transpose = mat.transpose();

    for (int i = 0; i < r; ++i)
    {
        for(unsigned j = 0; j < c; ++j) {
            EXPECT_EQ(transpose(j,i),mat(i,j));
        }
    }
}

TEST(Matrix,nonaligned){
    int r1=5;
    int c1=1;

    int r2 = 2;
    int c2 = 3;
    CL_Matrix<float> mat1(r1,c1);
    CL_Matrix<float> mat2(r2,c2);

    EXPECT_DEATH(mat1 += mat2,".*Assertion.*failed.*");

    EXPECT_DEATH(mat1 -= mat2,".*Assertion.*failed.*");


    EXPECT_DEATH(mat1.dot(mat2),".*Assertion.*failed.*");


}

TEST(Matrix,addsubmul){
    auto r=20u;
    auto c=20u;

    CL_Matrix<float> mat(r,c);
    CL_Matrix<float> other(r,c);

    mat.fill(5.);

    other.fill(3.);

    mat -= other;

    for(unsigned i = 0; i < r; ++i) {
        for(unsigned j = 0; j < c; ++j) {
            EXPECT_EQ(mat(i,j),2.);
        }
    }

    mat+=mat += mat;


    for(unsigned i = 0; i < r; ++i) {
        for(unsigned j = 0; j < c; ++j) {
            EXPECT_EQ(mat(i,j),8.);
        }
    }

    mat *= mat;


    for(unsigned i = 0; i < r; ++i) {
        for(unsigned j = 0; j < c; ++j) {
            EXPECT_EQ(mat(i,j),64.);
        }
    }

    CL_Matrix<float> addres = mat + mat;
    for(unsigned i = 0; i < r; ++i) {
		for(unsigned j = 0; j < c; ++j) {
			EXPECT_EQ(addres(i,j),128.);
		}
	}

    CL_Matrix<float> subres = mat - mat;
    for(unsigned i = 0; i < r; ++i) {
		for(unsigned j = 0; j < c; ++j) {
			EXPECT_EQ(subres(i,j),0.);
		}
	}
}

TEST(Matrix,mult){
	auto r=128u;
	auto c=128u;

	CL_Matrix<float> mat(r,c);
	mat.random(0,1);
	CL_Matrix<float> other(r,c);
	other.random(0,1);

	CL_Matrix<float> res = mat*other;
	for(unsigned i = 0; i < r; ++i) {
		for(unsigned j = 0; j < c; ++j) {
			EXPECT_NEAR(res(i,j),0.5,0.5);
		}
	}

}

TEST(OpenCL,Datatypes){
//	Testing all the different datatypes
    OpenCL a("kernels.cl") ;

    const int size = 15;
    std::vector<std::size_t> globsize = {size};
    std::vector<std::size_t> localsize = {1};
    float b[size] = {0};
    float c[size] = {0};

    std::vector<std::size_t> outputargs = {2};

    a.runKernel("sigmoid",outputargs,globsize,localsize,size,b,c);

    std::vector<float> d(size);
    std::vector<float> e(size);
    a.runKernel("sigmoid",outputargs,globsize,localsize,size,d,e);

    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(c[i],0.5);
        EXPECT_EQ(e[i],0.5);
    }

}

TEST(Matrix,dotsq){

    int r=512;
    int c=512;

    CL_Matrix<float> mat(r,c);
    CL_Matrix<float> other(c,r);
    mat.fill(5.);

    other.fill(2.);
    CL_Matrix<float> out = mat.dot(other);

    long count = 0;
    for(unsigned i = 0; i < r; ++i) {
        for(unsigned j = 0; j < r; ++j) {
            if (out(i,j)!= c*10){
                count ++ ;
            }
        }
    }
    EXPECT_EQ(count,0);

}

TEST(Matrix,dotuneven){

    int r=256;
    int c=512;

    CL_Matrix<float> mat(r,c);
    CL_Matrix<float> other(c,r);
    mat.fill(5.);

    other.fill(2.);
    CL_Matrix<float> out = mat.dot(other);

    long count = 0;
    for(unsigned i = 0; i < r; ++i) {
        for(unsigned j = 0; j < r; ++j) {
            if (out(i,j)!= c*10){
                count ++ ;
            }
        }
    }
    EXPECT_EQ(count,0);

}

TEST(Matrix,dotvector){

    auto r=1u;
    auto c=1024u;

    CL_Matrix<float> mat(r,c);
    CL_Matrix<float> other(c,r);
    mat.fill(5.);

    other.fill(2.);
    CL_Matrix<float> out = mat.dot(other);

    long count = 0;
    for(unsigned i = 0; i < r; ++i) {
        for(unsigned j = 0; j < r; ++j) {
            if (out(i,j)!= c*10){
                count ++ ;
            }
        }
    }
    EXPECT_EQ(count,0);

}


TEST(Matrix,dotvectordiff){

    auto r1=10u;
    auto c1=1u;

    auto r2=1u;
    auto c2=2u;

    CL_Matrix<float> mat(r1,c1);
    CL_Matrix<float> other(r2,c2);
    mat.fill(5.);

    other.fill(2.);
    CL_Matrix<float> out = mat.dot(other);
    long count = 0;
    for(unsigned i = 0; i < r1; ++i) {
        for(unsigned j = 0; j < c2; ++j) {
            if (out(i,j)!= 10.){
                count ++ ;
            }
        }
    }
    EXPECT_EQ(count,0);

}

TEST(Activation,Sigmoid){
    auto length = 10u;
    CL_Matrix<float> mat(length,1);
    mat.fill(0.);
	Sigmoid s;
	CL_Matrix<float> out = s.propagate(mat);
    for (auto i = 0u; i < length; ++i)
    {
        EXPECT_EQ(out(i,0),0.5);
    }

    CL_Matrix<float> mat2(length,length);

    mat2.fill(0.);
    CL_Matrix<float> out2 = s.propagate(mat2);
    for(auto i=0u; i < length ; i++){
    	for(auto j=0u; j < length ; j++){
    		EXPECT_EQ(out2(i,j),0.5);
		}
    }

}

TEST(Activation,Sigmoidgrad){
	auto size= 20u;
	CL_Matrix<float> f(size,size);

	CL_Matrix<float> res = f.sigmoidgrad();

	for(auto i=0u; i < size ; i++){
		for(auto j=0u; j < size ; j++){
			EXPECT_EQ(res(i,j),0.25);
		}
	}
}


TEST(Activation,Tanh){
    int length = 100;
    CL_Matrix<float> mat(length,1);
    mat.fill(1.);
	TanH tan;
	CL_Matrix<float> out = tan.propagate(mat);
    for (int i = 0; i < length; ++i)
    {
        EXPECT_NEAR(out(i,0),0.76,0.02);
    }
}


TEST(Nnet,feedforward){
	CL_Matrix<float> input(20,1);
	input.fill(0.);
	Sigmoid s;
	FeedForwardNN dnn(20,2,0.9);
	dnn.addActivation(&s);
	dnn.addActivation(&s);
	dnn.addHiddenLayer(10);
	CL_Matrix<float> out = dnn.feedforward(input);
}


TEST(Nnet,batchgradient){
	CL_Matrix<float> input(20,1);
	input.fill(0.);
	CL_Matrix<float> target(2,1);
	target.fill(1.);
	Sigmoid s;
	FeedForwardNN dnn(20,2,0.9);
	dnn.addActivation(&s);
	dnn.addActivation(&s);
	dnn.addHiddenLayer(10);
	dnn.trainbatch(input,target);
}


int main(int argc,char **argv){
    ::testing::InitGoogleTest(&argc, argv);
//	::testing::GTEST_FLAG(filter) = "Matrix.dotvectordiff:Matrix.dotvector";//":-:*Counter*";
    // Otherwise EXPECT_DEATH will warn us that the execution time may be too slow,
    // since EXPECT_DEATH uses forks, which could not be used
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    return RUN_ALL_TESTS();

}

