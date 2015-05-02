################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cost/Cost.cpp \
../src/cost/CrossEntropy.cpp \
../src/cost/MSE.cpp 

OBJS += \
./src/cost/Cost.o \
./src/cost/CrossEntropy.o \
./src/cost/MSE.o 

CPP_DEPS += \
./src/cost/Cost.d \
./src/cost/CrossEntropy.d \
./src/cost/MSE.d 


# Each subdirectory must supply rules for building sources it contributes
src/cost/%.o: ../src/cost/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


