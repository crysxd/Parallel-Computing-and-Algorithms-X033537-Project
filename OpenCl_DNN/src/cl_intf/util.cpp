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

