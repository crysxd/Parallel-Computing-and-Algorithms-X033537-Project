################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/activations/Activation.cpp \
../src/activations/Sigmoid.cpp \
../src/activations/TanH.cpp 

OBJS += \
./src/activations/Activation.o \
./src/activations/Sigmoid.o \
./src/activations/TanH.o 

CPP_DEPS += \
./src/activations/Activation.d \
./src/activations/Sigmoid.d \
./src/activations/TanH.d 


# Each subdirectory must supply rules for building sources it contributes
src/activations/%.o: ../src/activations/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


