#pragma OPENCL EXTENSION cl_intel_printf : enable

#define MAXSIZE 3400

typedef struct Node
{
    int numberOfWeights;
    float weights[MAXSIZE];
    float output;
    float input;
    float errorGradient;
} Node;

typedef struct HiddenLayer
{
    int numberOfNodes;
    Node nodes[MAXSIZE];
} HiddenLayer;


float inline sigmoid(float n)
{
    //To deal with overflow rounding errors and the such
    if (n < -100)
        return 0;
    if (n > 100)
        return 1;
    return 1/(1 + exp(-n));
}

// //Used to find the (row,nodeNumber) pair that corresponds to the n'th input/errorGradient node
 void inline getPosition(int n, constant int* restrict netsize, int* restrict row, int* restrict nodeNumber)
 {
     for (unsigned int i = 1; ;++i)//Termination is determined by the break statement
     {
         int k = netsize[i];
         bool comparison = k <= n;
         if (comparison)
             n += -k;
         else
         {
             *row = i;
             *nodeNumber = n;
             break;
         }
     }
 }
 
__kernel void setinput(__global HiddenLayer* restrict layers, constant float* inputs)
{
    const int i = get_global_id(0);
    layers[0].nodes[i].output = inputs[i];
}

 kernel void writeOutputToBuffer(global HiddenLayer* restrict layers, global float* restrict outputs, int lastLayer)
 {
     const int n = get_global_size(0);
     const int i = get_global_id(0);
     outputs[i] = layers[lastLayer].nodes[i].output;
 }


// Feedforward
__kernel void feedforward(__global HiddenLayer* restrict layers, constant int* restrict netsize)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0); //There will be an offset depending on the layer we are operating on

    int layer, nodeNumber, numberOfWeights;
    float t;
    getPosition(i, netsize, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;
    t = 0;
    for (unsigned int j = 0; j != numberOfWeights; ++j){
        t += layers[layer].nodes[nodeNumber].weights[j] * layers[layer-1].nodes[j].output;
        printf("Weight %.1f\n",layers[layer].nodes[nodeNumber].weights[j]);
    }
    printf("Layer %i node %i val %.1f\n",layer,nodeNumber,t);
    layers[layer].nodes[nodeNumber].output = sigmoid(t);
}
