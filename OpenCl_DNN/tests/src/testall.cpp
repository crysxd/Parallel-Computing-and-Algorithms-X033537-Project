#include "gtest/gtest.h"
#include "Activation.h"
#include "HiddenLayer.h"
#include "Sigmoid.h"
#include "TanH.h"


TEST(Activation,Sigmoid){
	Sigmoid s();
	HiddenLayer h(Sigmoid);
    EXPECT_EQ(1,1);
}

TEST(Activation,TanH){
	EXPECT_EQ(1,1)

}


int main(int argc,char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}

