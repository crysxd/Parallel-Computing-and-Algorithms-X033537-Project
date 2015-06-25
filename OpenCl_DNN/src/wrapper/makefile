all:
	g++ -std=c++11 -c -fPIC NeuralNetwork.cpp -o nn.o
	g++ -std=c++11 -shared -Wl,-soname,libnn.so -o libnn.so  nn.o

clean:
	rm nn.o libnn.so
