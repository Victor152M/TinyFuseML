#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


BUILD_DIR=build
PROJECT_ROOT=$(pwd)

echo "Cleaning previous build..."
rm -rf $BUILD_DIR

echo "Creating build directory..."
mkdir $BUILD_DIR
cd $BUILD_DIR

echo "Running CMake configuration..."
cmake -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 -Wno-dev ..

echo "Building project..."
cmake --build .

echo "Running Project..."
./tinyfuseml
