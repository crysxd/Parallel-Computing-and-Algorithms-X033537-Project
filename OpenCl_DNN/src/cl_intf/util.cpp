/*
 * util.cpp
 *
 *  Created on: May 3, 2015
 *      Author: hedi7
 */

#include "util.hpp"


char* util::file_contents(const char* filepath){
    ifstream readstr(filepath);
    string line;
    stringstream strstream;
    strstream << readstr.rdbuf();
    string retstr = strstream.str();
    char *chr = new char[retstr.length() + 1];
    strcpy (chr,retstr.c_str());

    return chr;

}


template<typename T>
inline void util::randinit(int min, int max, std::vector<T> arr) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(min,max);

	for(int i = 0 ; i < arr.size() ; i ++){
		arr[i]=dist(mt);
	}

}

float util::randfloat(int min, int max) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(min,max);
	return dist(mt);
}
