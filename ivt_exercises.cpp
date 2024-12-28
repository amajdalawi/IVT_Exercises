// Image and Video Technology, Colas Schretter <colas.schretter@vub.be>
// This example program compares the C syntax for linear and multidimensional arrays
// Compilation: g++ -Wall -Wextra -pedantic -o ivt ivt_exercises.cpp
#include <random>
#include <utility> // For std::pair

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
        if (!file) {
            std::cerr << "Error: Failed to read image data at row " << i << " from file." << std::endl;
            for (int j = 0; j <= i; ++j) {
                delete[] image[j];
            }
            delete[] image;
            return nullptr;
        }
    }

    file.close();
    std::cout << "Image successfully loaded from " << filename << std::endl;

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

// Function to compute the Mean Squared Error (MSE) between two images
float mse(float** image1, float** image2) {
    float mse = 0.0f;
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float diff = image1[y][x] - image2[y][x];
            mse += diff * diff;
        }
    }
    mse /= (WIDTH * HEIGHT);
    return mse;
}

// Function to compute the Peak Signal-to-Noise Ratio (PSNR) between two images
float psnr(float** image1, float** image2, float max) {
    float mse_val = mse(image1, image2);
    if (mse_val == 0) {
        return INFINITY; // No error implies infinite PSNR
    }
    return 10.0f * log10((max * max) / mse_val);
}

float** generateUniformNoise(float minVal, float maxVal) {
    float** image = new float* [HEIGHT];
    for (int i = 0; i < HEIGHT; ++i) {
        image[i] = new float[WIDTH];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(minVal, maxVal);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            image[y][x] = dist(gen);
        }
    }

    return image;
}

// Function to generate a Gaussian distributed random noise image
float** generateGaussianNoise(float mean, float stddev) {
    float** image = new float* [HEIGHT];
    for (int i = 0; i < HEIGHT; ++i) {
        image[i] = new float[WIDTH];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            image[y][x] = dist(gen);
        }
    }

    return image;
}

// Function to calculate the mean and variance of an image
std::pair<float, float> calculateMeanAndVariance(float** image) {
    float mean = 0.0f;
    float variance = 0.0f;
    int totalPixels = WIDTH * HEIGHT;

    // Calculate mean
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            mean += image[y][x];
        }
    }
    mean /= totalPixels;

    // Calculate variance
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float diff = image[y][x] - mean;
            variance += diff * diff;
        }
    }
    variance /= totalPixels;

    return std::make_pair(mean, variance);
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

    // SESSION 1 PART 2

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

    // SESSION 1 PART 3

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


    // SESSION 2 PART 1

    float** original_image = load("parrot_256x256.raw");
    float** blurred_image = load("blurred.raw");
    float** sharpened_image = load("sharpened_from_adding.raw");

    float mse_sharpened = mse(original_image, sharpened_image);
    float mse_blurred = mse(original_image, blurred_image);
    float mse_normal = mse(original_image, original_image);
    std::cout << "Mean Squared Error Sharpened (MSE): " << mse_sharpened << std::endl;
    std::cout << "Mean Squared Error Normal (MSE): " << mse_normal << std::endl;
    std::cout << "Mean Squared Error Blurred (MSE): " << mse_blurred << std::endl;

    float psnr_blurred = psnr(original_image, blurred_image, 255);
    float psnr_normal = psnr(original_image, original_image, 255);
    float psnr_sharpened = psnr(original_image, sharpened_image, 255);
    std::cout << "psnr blurred: " << psnr_blurred << std::endl;
    std::cout << "psnr sharpened: " << psnr_sharpened << std::endl;
    std::cout << "psnr normal: " << psnr_normal << std::endl;

    // SESSION 2 PART 2
        // Generate uniform noise
    float** uniformNoise = generateUniformNoise(-0.5f, 0.5f);
    const char* uniformFilename = "uniform_noise.raw";
    store(uniformFilename, uniformNoise);

    // Generate Gaussian noise
    float** gaussianNoise = generateGaussianNoise(0.0f, 0.1443f); // Variance equivalent to uniform noise
    const char* gaussianFilename = "gaussian_noise.raw";
    store(gaussianFilename, gaussianNoise);


    // Calculate mean and variance for uniform noise
    std::pair<float, float> uniformStats = calculateMeanAndVariance(uniformNoise);
    std::cout << "Uniform Noise - Mean: " << uniformStats.first << ", Variance: " << uniformStats.second << std::endl;

    // Calculate mean and variance for Gaussian noise
    std::pair<float, float> gaussianStats = calculateMeanAndVariance(gaussianNoise);
    std::cout << "Gaussian Noise - Mean: " << gaussianStats.first << ", Variance: " << gaussianStats.second << std::endl;



	return EXIT_SUCCESS;
}