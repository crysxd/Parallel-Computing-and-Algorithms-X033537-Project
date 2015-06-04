#include "gtest/gtest.h"
#include "Activation.h"
#include "HiddenLayer.h"
#include "Sigmoid.h"
#include "TanH.h"

#include <vector>
#include "util.hpp"
#include "OpenCL.hpp"
#include "CLMatrix.hpp"

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
    int r=10;
    int c=5;
    CL_Matrix<float> mat1(10,5);
    CL_Matrix<float> mat2(8,5);

    EXPECT_DEATH(mat1 += mat2,".*Assertion.*failed.*");

    EXPECT_DEATH(mat1 -= mat2,".*Assertion.*failed.*");

    EXPECT_DEATH(mat1.dot(mat2),".*Assertion.*failed.*");


}

TEST(Matrix,addsubmul){
    int r=100;
    int c=100;

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
}

TEST(TEST,CL){
    OpenCL a("sigmoid.cl") ;
    std::vector<float> b(100);

    int i =4;

    a.runKernel("sigmoid",12,b,i);
    // float a[1000];
    // for (int i = 0; i < 1000; ++i)
    // {
    //     a[i] = (float)i;
    // }
    // cl_intf.runKernel("sigmoid",15,23,b);
    // Tuple a;
    // a.iter(1.0,2);
    // TestCl<float,float> a(1.0,1.0);
}

TEST(Matrix,dot){

    int r=10;
    int c=30;

    CL_Matrix<float> mat(r,c);
    CL_Matrix<float> other(c,r);
    mat.fill(5.);

    other.fill(2.);
    mat.dot(other);

    // for(unsigned i = 0; i < r; ++i) {
    //     for(unsigned j = 0; j < c; ++j) {
    //         EXPECT_EQ()
    //     }
    // }

}


TEST(Activation,Sigmoid){
    int length = 10;
    std::vector<float> v;
    for (int i = 0; i < length; ++i)
    {
        v.push_back(0);
    }
	Sigmoid s;
    std::vector<float> ret= s.activate(v);
    for (int i = 0; i < ret.size(); ++i)
    {
        EXPECT_EQ(ret[i],0.5);
    }
}

TEST(Activation,SigmoidDouble){
    int testlength =10;
    std::vector<float> v;
    for (int i = 0; i < testlength; ++i)
    {
        v.push_back(0);
    }
    Sigmoid s;
    std::vector<float> ret= s.activate(v);
    EXPECT_EQ(ret.size(), testlength);
    for (int i = 0; i < ret.size(); ++i)
    {
        EXPECT_EQ(ret[i],0.5);
    }
    ret = s.activate(ret);
    EXPECT_EQ(ret.size() , testlength);
    for(unsigned i = 0; i < ret.size(); ++i) {
        EXPECT_NEAR(ret[i],0.62,0.01);
    }
}

TEST(Activation,SigmoidGPUChain){
    int length = 100000;
    std::vector<float> v;
    for (int i = 0; i < length; ++i)
    {
        v.push_back(0);
    }
    Sigmoid s;
    std::vector<util::GPU_Buffer> gpu_res= s.activateKeep(v);
    std::vector<float> res = s.activate(gpu_res);

    // Activate one more
    EXPECT_EQ(res.size() , length);
    for (int i = 0; i < res.size(); ++i)
    {
        EXPECT_NEAR(res[i],0.62,0.01);
    }

}


TEST(HiddenLayer,Activate){
    Sigmoid s;
    HiddenLayer h(&s,100);

}

TEST(Activation,TanH){

	std::vector<float> v;
    for (int i = 0; i < 10000; ++i)
    {
        v.push_back(0);
    }
}




int main(int argc,char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    // Otherwise EXPECT_DEATH will warn us that the execution time may be too slow,
    // since EXPECT_DEATH uses forks, which could not be used
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    return RUN_ALL_TESTS();

}

