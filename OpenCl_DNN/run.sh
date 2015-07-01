if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ../
make -j4
cd ../demo
python nnet.py train 784 1024 10 -ep 100 -N 100
