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
cmake ..

echo "Building project..."
cmake --build .

echo "Running Project..."
./cuda_project

