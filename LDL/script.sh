# this will create a c1.c file - the C source code to build a python extension
cython $1.pyx
# Compile the object file   
gcc -c -O3 -fPIC -I/usr/include/python2.7/ $1.c
# Link it into a shared library
gcc -shared $1.o -o $1.so
rm $1.c
rm $1.o
