// Image and Video Technology, Colas Schretter <colas.schretter@vub.be>
// This example program compares the C syntax for linear and multidimensional arrays
// Compilation: g++ -Wall -Wextra -pedantic -o ivt ivt_exercises.cpp

#include<fstream>
#include<cmath>
#include <iostream>
using std::cout;
using std::endl;

// storing images in memory with a one-dimensional array
float *create_image(const int height, const int width) {
    // dynamic memory allocation of a one-dimensional array
    float *image = new float[height * width];

    // fill in some values
    for(int y = 0; y < height; y++)
        for(int x = 0; x < 256; x++)
            image[y * width + x] = x + y * width;

    // return the pointer to the first image element
    return image;
}

// storing images in memory with a two-dimensional array
float (*create_image(const int height))[256] {
    // dynamic memory allocation of a two-dimensional array
    float (*image)[256] = new float[height][256];

    // fill in some values
    for(int y = 0; y < height; y++)
        for(int x = 0; x < 256; x++)
            image[y][x] = x + y * 256;

    // return the pointer to the first image element
    return image;
}


// Constants for image dimensions
const int WIDTH = 256;
const int HEIGHT = 256;

// Function to generate the cosine pattern
float** generateCosinePattern() {
    float** image = new float* [HEIGHT];
    for (int i = 0; i < HEIGHT; ++i) {
        image[i] = new float[WIDTH];
    }

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            image[y][x] = 0.5f + 0.5f * cos(x * M_PI / 32) * cos(y * M_PI / 64);
        }
    }

    return image;
}

// Function to store the image in RAW format
void store(const char* filename, float** image) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    for (int y = 0; y < HEIGHT; ++y) {
        file.write(reinterpret_cast<const char*>(image[y]), WIDTH * sizeof(float));
    }
    file.close();
    if (file) {
        std::cout << "Image successfully saved to " << filename << std::endl;
    }
    else {
        std::cerr << "Error: Failed to write image to file." << std::endl;
    }
}

// Function to load a RAW image into a 2D array
float** load(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return nullptr;
    }

    float** image = new float* [HEIGHT];
    for (int i = 0; i < HEIGHT; ++i) {
        image[i] = new float[WIDTH];
        file.read(reinterpret_cast<char*>(image[i]), WIDTH * sizeof(float));
    }

    file.close();
    if (file) {
        std::cerr << "Error: Failed to read image from file." << std::endl;
    }
    else {
        std::cout << "Image successfully loaded from " << filename << std::endl;
    }

    return image;
}

// Function to multiply two images pixel-by-pixel
float** multiplyImages(float** image1, float** image2) {
    float** result = new float* [HEIGHT];
    for (int i = 0; i < HEIGHT; ++i) {
        result[i] = new float[WIDTH];
    }

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            result[y][x] = image1[y][x] * image2[y][x];
        }
    }

    return result;
}


int main() {
    // create a two-dimensional array of 320x256 pixels
    float (*image2D)[256] = create_image(320);

    // create a one-dimensional array containing 81920 (320x256) pixels
    float *image1D = create_image(320,256);

    // access to a specific pixel using the 2D and 1D arrays
    const int y = 7, x = 5;
    cout << "the value of element at row number " << y << " and column number " << x << " is " << image2D[y][x] << endl;
    cout << "the value of element at row number " << y << " and column number " << x << " is " << image1D[y * 256 + x] << endl;

    // free the dynamic memory allocations
    delete [] image2D;
    delete [] image1D;

    // Generate the cosine pattern
    float** cosinePattern = generateCosinePattern();

    // Store the cosine pattern to a RAW file
    const char* cosineFilename = "cosine_pattern.raw";
    store(cosineFilename, cosinePattern);

    // Load the parrot image
    const char* parrotFilename = "parrot_256x256.raw";
    float** parrotImage = load(parrotFilename);
    if (!parrotImage) {
        // Clean up cosine pattern memory if loading failed
        for (int i = 0; i < HEIGHT; ++i) {
            delete[] cosinePattern[i];
        }
        delete[] cosinePattern;
        return 1;
    }

    // Multiply the parrot image with the cosine pattern
    float** modifiedImage = multiplyImages(parrotImage, cosinePattern);

    // Store the modified image to a RAW file
    const char* modifiedFilename = "modified_parrot.raw";
    store(modifiedFilename, modifiedImage);

    // Clean up dynamically allocated memory
    for (int i = 0; i < HEIGHT; ++i) {
        delete[] cosinePattern[i];
        delete[] parrotImage[i];
        delete[] modifiedImage[i];
    }
    delete[] cosinePattern;
    delete[] parrotImage;
    delete[] modifiedImage;




	return EXIT_SUCCESS;
}
