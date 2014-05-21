/*
 * mymodule.cpp
 *
 *  Created on: May 21, 2014
 *      Author: Jatin
 */
#include "main.cpp"

#include <boost/python.hpp>
BOOST_PYTHON_MODULE(mymodule)
{
    def("detect",detect);
}

