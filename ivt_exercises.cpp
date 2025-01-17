// Image and Video Technology, Colas Schretter <colas.schretter@vub.be>
// This example program compares the C syntax for linear and multidimensional arrays
// Compilation: g++ -Wall -Wextra -pedantic -o ivt ivt_exercises.cpp
#include <random>
#include <utility> // For std::pair
#include <cstring>
#include<fstream>
#include<cmath>
#include <iostream>
using std::cout;
using std::endl;

std::random_device rd;
std::mt19937 gen(rd());


const int ZIGZAG_ORDER[64][2] = {
    {0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2},
    {2, 1}, {3, 0}, {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4}, {0, 5},
    {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}, {6, 0}, {5, 1}, {4, 2},
    {3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4},
    {4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1}, {6, 2}, {5, 3}, {4, 4},
    {3, 5}, {2, 6}, {1, 7}, {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3},
    {7, 2}, {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7}, {4, 7}, {5, 6},
    {6, 5}, {7, 4}, {7, 5}, {6, 6}, {5, 7}, {6, 7}, {7, 6}, {7, 7}
};



 float QTable[8][8] = {
        {16.0f, 11.0f, 10.0f, 16.0f, 24.0f, 40.0f, 51.0f, 61.0f},
        {12.0f, 12.0f, 14.0f, 19.0f, 26.0f, 58.0f, 60.0f, 55.0f},
        {14.0f, 13.0f, 16.0f, 24.0f, 40.0f, 57.0f, 69.0f, 56.0f},
        {14.0f, 17.0f, 22.0f, 29.0f, 51.0f, 87.0f, 80.0f, 62.0f},
        {18.0f, 22.0f, 37.0f, 56.0f, 68.0f, 109.0f, 103.0f, 77.0f},
        {24.0f, 35.0f, 55.0f, 64.0f, 81.0f, 104.0f, 113.0f, 92.0f},
        {49.0f, 64.0f, 78.0f, 87.0f, 103.0f, 121.0f, 120.0f, 101.0f},
        {72.0f, 92.0f, 95.0f, 98.0f, 112.0f, 100.0f, 103.0f, 99.0f}
};


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
void store(const char* filename, float** image, int height, int width) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    for (int y = 0; y < height; ++y) {
        file.write(reinterpret_cast<const char*>(image[y]), width * sizeof(float));
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

    for (int x = 0; x < HEIGHT; ++x) {
        for (int y = 0; y < WIDTH; ++y) {
            result[x][y] = image1[x][y] * image2[x][y];
        }
    }

    return result;
}

// Function to compute the Mean Squared Error (MSE) between two images
float mse(float** image1, float** image2) {
    float mse = 0.0f;
    for (int x = 0; x < HEIGHT; ++x) {
        for (int y = 0; y < WIDTH; ++y) {
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

    std::uniform_real_distribution<float> dist(minVal, maxVal);

    for (int x = 0; x < HEIGHT; ++x) {
        for (int y = 0; y < WIDTH; ++y) {
            image[x][y] = dist(gen);
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
    for (int x = 0; x < HEIGHT; ++x) {
        for (int y = 0; y < WIDTH; ++y) {
            mean += image[x][y];
        }
    }
    mean /= totalPixels;

    // Calculate variance
    for (int x = 0; x< HEIGHT; ++x) {
        for (int y = 0; y < WIDTH; ++y) {
            float diff = image[x][y] - mean;
            variance += diff * diff;
        }
    }
    variance /= totalPixels;

    return std::make_pair(mean, variance);
}

float** getImageWithAddedNoise(float** image, float MSE) {
    cout << "entering function";
    /*float** noise = new float* [256];
    for (int i = 0; i < 256; ++i) {
        noise[i] = new float[256];
    }*/
    float ** noise = generateGaussianNoise(0, std::sqrt(MSE));
    float** newImage = new float* [256];
    for (int i = 0; i < 256; ++i) {
        newImage[i] = new float[256];
    }

    //std::random_device rd;
    //std::mt19937 gen(rd());
    //std::normal_distribution<float> dist(0.0f, std::sqrt(MSE));
    //cout << "hmm";
    ////cout << image[4][4];
    //cout << "ok";
    //for (int x = 0; x < 256; ++x) {
    //    for (int y = 0; y < 256; ++y) {
    //        noise[x][y] = dist(gen);
    //        //cout << noise[x][y] << " ";
    //    }
    //    //cout << endl;
    //}
    //for (int x = 0; x < 256; ++x) {
    //    for (int y = 0; x < 256; ++y) {
    //        newImage[x][y] =0.0f;
    //        newImage[x][y] = image[x][y] + noise[x][y];
    //        //cout << newImage[x][y] << " ";
    //    }
    //    cout << "ok we are here: " << x << endl;

    //}
    cout << "we are out!";
    //cout << noise[0][5];


    for (int x = 0; x < 256; ++x) {
        for (int y = 0; y < 256; ++y) {
            //cout << noise[x][y] ;
            //cout << x << " " << y << endl;
            newImage[x][y] = noise[x][y] + image[x][y];
            //cout << newImage[x][y];
        }
    }
    cout << "BRO" << endl;
    return newImage;


}



// Session 3 part 1

float** createDctMatrix() {
    float** matrix = new float* [WIDTH];
    for (int x = 0; x < WIDTH; x++) {
        matrix[x] = new float[HEIGHT];
    }

    for (int i = 0; i <WIDTH; i++) {
        for (int y = 0; y < HEIGHT; y++) {
            matrix[i][y] = cos(i * M_PI / 256 * (y + 0.5));
        }
    }

    return matrix;
}


// function to normalize a matrix, with a given row and column numbers
void normalizeMatrix(float** matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        float total = 0;
        for (int j = 0; j < rows; j++) {
            total += matrix[i][j] * matrix[i][j];
        }
        float normalization = 1 / sqrt(total);
        for (int j = 0; j < rows; j++) {
            matrix[i][j] *= normalization;
        }

    }
}

// helper print function
void print2DArray(float** arr, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << arr[i][j] << " ";
        }
        cout << endl; // Move to the next line after each row
    }
}

// Function to transpose a matrix
float** transpose(float** matrix, int rows, int cols) {
    float** transposed = new float* [cols];
    for (int i = 0; i < cols; ++i) {
        transposed[i] = new float[rows];
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}


// Function to multiply two images pixel-by-pixel
//float** multiplyMatricies(float** image1, float** image2) {
//    float** result = new float* [HEIGHT];
//    for (int i = 0; i < HEIGHT; ++i) {
//        result[i] = new float[WIDTH];
//    }
//
//    for (int y = 0; y < HEIGHT; ++y) {
//        for (int x = 0; x < WIDTH; ++x) {
//            result[y][x] = image1[y][x] * image2[y][x];
//        }
//    }
//
//    return result;
//}

float** multiplyMatrices(float** matrix1, int rows1, int cols1, float** matrix2, int rows2, int cols2) {
    if (cols1 != rows2) {
        std::cerr << "Error: Matrix dimensions do not allow multiplication." << std::endl;
        return nullptr;
    }

    float** result = new float* [rows1];
    for (int i = 0; i < rows1; ++i) {
        result[i] = new float[cols2];
        for (int j = 0; j < cols2; ++j) {
            result[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

// Session 3 part 2 (or part 8) 
// Function to extract a specific row from a 2D matrix
float* extractRow(float** matrix, int row, int cols) {
    float* extractedRow = new float[cols];
    for (int i = 0; i < cols; ++i) {
        extractedRow[i] = matrix[row][i];
    }
    return extractedRow;
}

// Function to store a single row in RAW format
void storeRawRow(const char* filename, float* row, int length) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing row: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(row), length * sizeof(float));
    file.close();
    if (file) {
        std::cout << "Row successfully saved to " << filename << std::endl;
    }
    else {
        std::cerr << "Error: Failed to write row to file." << std::endl;
    }
}

// Function to compute the DCT transform for a row
float* transformRow(float* row, float** dctBasis, int length) {
    float* dctCoefficients = new float[length];
    for (int k = 0; k < length; ++k) {
        dctCoefficients[k] = 0.0f;
        for (int x = 0; x < length; ++x) {
            dctCoefficients[k] += row[x] * dctBasis[k][x];
        }
    }
    return dctCoefficients;
}

float* restoreRow(float* row, float** transposedDctBasis, int length) {
    float* restoredVals = new float[length];
    for (int k = 0; k < length; ++k) {
        restoredVals[k] = 0.0f;
        for (int x = 0; x < length; ++x) {
            restoredVals[k] += row[x] * transposedDctBasis[k][x];
        }
    }

    return restoredVals;
}

float* thresholdCoefficients(float* dctCoefficients, int length, float threshold) {
    float* thresholdedCoefficients = new float[length];
    int count = 0;
    for (int i = 0; i < length; ++i) {
        if (std::abs(dctCoefficients[i]) < threshold) {
            thresholdedCoefficients[i] = 0.0f;
            count++;
        }
        else {
            thresholdedCoefficients[i] = dctCoefficients[i];
        }
    }
    std::cout << " The number  thresholded values for this signal is: " << count << std::endl;
    return thresholdedCoefficients;
}


void printRow(float* someRow, int length) {
    for (int i = 0; i < length; i++) {
        cout << someRow[i] << " " << endl;
    }
    cout << "\n" << endl;
}

// Function to compute MSE and PSNR between two rows
float psnrRow(float* originalRow, float* restoredRow, int length, float maxValue) {
    float mse = 0.0f;
    for (int i = 0; i < length; ++i) {
        float diff = originalRow[i] - restoredRow[i];
        mse += diff * diff;
    }
    mse /= length;

    if (mse == 0) {
        return INFINITY; // No error implies infinite PSNR
    }
    
    float maxPixelValue = maxValue; // Assuming normalized pixel values in [0, 1]
    float psnr = 10.0f * log10((maxPixelValue * maxPixelValue) / mse);
    return psnr;
}


// session 4
float** transform2D(float** image, float** dctMatrix) {

    // multiply A times the image X
    float** intermediateStep = multiplyMatrices(dctMatrix, 256, 256, image, 256, 256);
    // tranpose the result
    float** firstTranspose = transpose(intermediateStep, 256, 256);
    // multiply A times the transposed result
    float** secondStep = multiplyMatrices(dctMatrix, 256, 256, firstTranspose, 256, 256);
    // now transpose it again and return it
    float** secondTranspose = transpose(secondStep, 256, 256);
    return secondTranspose;
}

float** restoreTransform2d(float** coefficients, float** idctMatrix, float** dctMatrix) {
    // multiply A^T times the coeffieicnts Y
    float** firstStep = multiplyMatrices(idctMatrix, 256, 256, coefficients, 256, 256);
    // now multiply the result of the first Step (call it C) by the original dct matrix A
    float** secondStep = multiplyMatrices(firstStep, 256, 256, dctMatrix, 256, 256);
    return secondStep;
}

float** threshold2D(float** dctCoefficients, int size, float threshold) {
    float** thresholdedCoefficients = new float* [HEIGHT];
    for (int i = 0; i < size; i++) {
         thresholdedCoefficients[i] = new float[HEIGHT];
    }
    int count = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; j++) {
            if (std::abs(dctCoefficients[i][j]) < threshold) {
                thresholdedCoefficients[i][j] = 0.0f;
                count++;
            }
            else {
                thresholdedCoefficients[i][j] = dctCoefficients[i][j];
            }
        }
        
        
    }
    std::cout << " The number  thresholded values for this signal is: " << count << std::endl;
    return thresholdedCoefficients;
}

//float psnrMatrix(float* originalRow, float* restoredRow, int length, float maxValue) {
//    float mse = 0.0f;
//    for (int i = 0; i < length; ++i) {
//        float diff = originalRow[i] - restoredRow[i];
//        mse += diff * diff;
//    }
//    mse /= length;
//
//    if (mse == 0) {
//        return INFINITY; // No error implies infinite PSNR
//    }
//
//    float maxPixelValue = maxValue; // Assuming normalized pixel values in [0, 1]
//    float psnr = 10.0f * log10((maxPixelValue * maxPixelValue) / mse);
//    return psnr;
//}



// Session 04 Part 10
float** generate8Dct() {
    float** matrix = new float* [8];
    for (int x = 0; x < 8; x++) {
        matrix[x] = new float[8];
    }

    for (int i = 0; i < 8; i++) {
        for (int y = 0; y < 8; y++) {
            matrix[i][y] = cos(i * M_PI / 8 * (y + 0.5));
        }
    }
    normalizeMatrix(matrix, 8, 8);
    return matrix;

}


float** applyDCT(float** block, float** dctMatrix) {
    float** intermediateStep = multiplyMatrices(dctMatrix, 8, 8, block, 8, 8);
    float** firstTranspose = transpose(intermediateStep, 8, 8);
    float** secondStep = multiplyMatrices(dctMatrix, 8, 8, firstTranspose, 8, 8);
    float** secondTranspose = transpose(secondStep, 8, 8);
    return secondTranspose;

}

float** applyQ(float** transformedBlock) {
    float** quantizedBlock = new float* [8];
    for (int i = 0; i < 8; i++) {
        quantizedBlock[i] = new float[8];
        for (int j = 0; j < 8; j++) {
            quantizedBlock[i][j] = std::round(transformedBlock[i][j] / QTable[i][j]);
        }
    }
    return quantizedBlock;
}

float** applyIQ(float** quantizedBlock) {
    float** reversedQBlock = new float* [8];
    for (int i = 0; i < 8; i++) {
        reversedQBlock[i] = new float[8];
        for (int j = 0; j < 8; j++) {
            reversedQBlock[i][j] = quantizedBlock[i][j] * QTable[i][j];
        }
    }
    return reversedQBlock;
}

float** applyIDCT(float** reverseQBlock, float** idct, float** dct) {
    float** firstStep = multiplyMatrices(idct, 8, 8, reverseQBlock, 8, 8);
    float** secondStep = multiplyMatrices(firstStep, 8, 8, dct, 8, 8);
    return secondStep;
}

void approximate(float** image, int size) {

    // Generate DCT and IDCT matrices
    float** dctMatrix = generate8Dct();
    float** idctMatrix = transpose(dctMatrix,8,8);

    // Create intermediate matrices for storing results
    float** dctCoefficients = new float* [size];
    float** quantizedCoefficients = new float* [size];
    float** reverseQuantized = new float* [size];
    float** restoredImage = new float* [size];
    for (int i = 0; i < size; ++i) {
        dctCoefficients[i] = new float[size];
        quantizedCoefficients[i] = new float[size];
        restoredImage[i] = new float[size];
        reverseQuantized[i] = new float[size];
        //memset(dctCoefficients[i], 0, size * sizeof(float));
        //memset(quantizedCoefficients[i], 0, size * sizeof(float));
        //memset(reverseQuantized[i], 0, size * sizeof(float));
        //memset(restoredImage[i], 0, size * sizeof(float));
        for (int j = 0; j < 256; j++) {
            dctCoefficients[i][j] = 0.0f;
            quantizedCoefficients[i][j] = 0.0f;
            reverseQuantized[i][j] = 0.0f;
            restoredImage[i][j] = 0.0f;
        }
    }

    // Process each 8x8 block
    for (int row = 0; row < size; row += 8) {
        for (int col = 0; col < size; col += 8) {
            // Extract 8x8 block
            float** block = new float* [8];
            for (int i = 0; i < 8; ++i) {
                block[i] = new float[8];
                for (int j = 0; j < 8; ++j) {
                    block[i][j] = image[row + i][col + j];
                }
            }

            // Apply DCT
            float** dctBlock = applyDCT(block, dctMatrix);

            // Quantize
            float** quantizedBlock = applyQ(dctBlock);

            // Inverse quantize
            float** iqBlock = applyIQ(quantizedBlock);

            // Apply IDCT
            float** restoredBlock = applyIDCT(iqBlock, idctMatrix, dctMatrix);

            // Save results back to the image matrices
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    dctCoefficients[row + i][col + j] = dctBlock[i][j];
                    quantizedCoefficients[row + i][col + j] = quantizedBlock[i][j];
                    restoredImage[row + i][col + j] = restoredBlock[i][j];
                    reverseQuantized[row + i][col + j] = iqBlock[i][j];
                }
            }

            // Clean up block memory
            for (int i = 0; i < 8; ++i) {
                delete[] block[i];
                delete[] dctBlock[i];
                delete[] quantizedBlock[i];
                delete[] iqBlock[i];
                delete[] restoredBlock[i];
            }
            delete[] block;
            delete[] dctBlock;
            delete[] quantizedBlock;
            delete[] iqBlock;
            delete[] restoredBlock;
        }
    }

    // Save intermediate results
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part10_coefficnet_image.raw", dctCoefficients, size, size);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part10_Quantized_image.raw", quantizedCoefficients, size, size);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part10_restoredImage.raw", restoredImage, size, size);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part10_IQ_image.raw", reverseQuantized, 256, 256);
}

float** encode(float** image, int size) {
    float** dctMatrix = generate8Dct();

    float** quantizedCoefficients = new float* [size];
    for (int i = 0; i < size; ++i) {
        quantizedCoefficients[i] = new float[size];
        for (int j = 0; j < 256; j++) {
            quantizedCoefficients[i][j] = 0.0f;
        }
    }

    for (int row = 0; row < size; row += 8) {
        for (int col = 0; col < size; col += 8) {
            // Extract 8x8 block
            float** block = new float* [8];
            for (int i = 0; i < 8; ++i) {
                block[i] = new float[8];
                for (int j = 0; j < 8; ++j) {
                    block[i][j] = image[row + i][col + j];
                }
            }

            // Apply DCT
            float** dctBlock = applyDCT(block, dctMatrix);

            // Quantize
            float** quantizedBlock = applyQ(dctBlock);
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    quantizedCoefficients[row + i][col + j] = quantizedBlock[i][j];
                }
            }
        }
    }

    return quantizedCoefficients;
}

float** decode(float** quantizedImage, int size) {
    float** dctMatrix = generate8Dct();
    float** idctMatrix = transpose(dctMatrix, 8, 8);
    float** restoredImage = new float* [size];
    for (int i = 0; i < size; ++i) {
        restoredImage[i] = new float[size];
        for (int j = 0; j < 256; j++) {
            restoredImage[i][j] = 0.0f;
        }
    }
    for (int row = 0; row < size; row += 8) {
        for (int col = 0; col < size; col += 8) {
            // Extract 8x8 block
            float** block = new float* [8];
            for (int i = 0; i < 8; ++i) {
                block[i] = new float[8];
                for (int j = 0; j < 8; ++j) {
                    block[i][j] = quantizedImage[row + i][col + j];
                }
            }


            // Inverse quantize
            float** iqBlock = applyIQ(block);

            // Apply IDCT
            float** restoredBlock = applyIDCT(iqBlock, idctMatrix, dctMatrix);

            // Save results back to the image matrices
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    restoredImage[row + i][col + j] = restoredBlock[i][j];
                }
            }

            // Clean up block memory
            for (int i = 0; i < 8; ++i) {
                delete[] block[i];
                delete[] iqBlock[i];
                delete[] restoredBlock[i];
            }
            delete[] block;
            delete[] iqBlock;
            delete[] restoredBlock;
        }
    }
    return restoredImage;
}

// Session 05 Part 12
float** getDC(float** image) {
    float** dcImage = new float* [32];
    for (int i = 0; i < 32; i++) {
        dcImage[i] = new float[32];
    }
    for (int i = 0; i < 256; i += 8) {
        for (int j = 0; j < 256; j += 8) {
            dcImage[i / 8][j / 8] = image[i][j];
        }
    }

    return dcImage;
}


// Function to delta encode DC terms
void deltaEncodeDC(float** dcImage, int size, const char* filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file for writing delta encoded DC terms." << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cout << dcImage[i][j] << " ";
            float delta = 0.0f;
            if (i == 0 && j == 0) {
                delta = dcImage[i][j]; // First term
            }
            else if (j == 0) {
                delta = dcImage[i][j] - dcImage[i - 1][size - 1]; // First in row
            }
            else {
                delta = dcImage[i][j] - dcImage[i][j - 1]; // Subsequent in row
            }
            file << delta << " ";
        }
        cout << "\n";
        file << "\n";
    }

    file.close();
}

// Function to decode delta encoded DC terms
float** deltaDecodeDC(const char* filename, int size) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file for reading delta encoded DC terms." << std::endl;
        return nullptr;
    }

    float** dcImage = new float* [size];
    for (int i = 0; i < size; ++i) {
        dcImage[i] = new float[size];
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float delta;
            file >> delta;
            if (i == 0 && j == 0) {
                dcImage[i][j] = delta; // First term
            }
            else if (j == 0) {
                dcImage[i][j] = delta + dcImage[i - 1][size - 1]; // First in row
            }
            else {
                dcImage[i][j] = delta + dcImage[i][j - 1]; // Subsequent in row
            }
        }
    }
    file.close();
    return dcImage;
}

// Session 05 part 13

// Function to generate RLE AC coefficients for an 8�8 block
std::vector<std::pair<int, float>> generateRLEAC(float block[8][8]) {
    std::vector<std::pair<int, float>> rleAC;
    int zeroCount = 0;

    // Start from the second element in zigzag order for ac coeffs.
    for (int i = 1; i < 64; ++i) {
        int row = ZIGZAG_ORDER[i][0];
        int col = ZIGZAG_ORDER[i][1];
        float value = block[row][col];

        if (value == 0.0f) {
            ++zeroCount;
        }
        else {
            // Push the run-length and the non-zero value to the vector
            rleAC.emplace_back(zeroCount, value);
            zeroCount = 0; // Reset zero count
        }
    }

    // Add end-of-block marker (0, 0) if there are trailing zeros
    if (rleAC.empty() || zeroCount > 0) {
        rleAC.emplace_back(0, 0.0f);
    }

    return rleAC;
}

// Function to write RLE AC coefficients to a file
void writeRLEToFile(const std::vector<std::pair<int, float>>& rleAC, std::ofstream& outFile) {
    for (const auto& pair : rleAC) {
        outFile << pair.first << " " << pair.second << " ";
        //std::cout << pair.first << " " << static_cast<int>(pair.second);
    }
    outFile << std::endl;
}

// Function to encode a DCT-quantized image of size 256�256 into RLE AC coefficients
void encodeRLE(float** quantizedImage, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return;
    }

    for (int row = 0; row < 256; row += 8) {
        for (int col = 0; col < 256; col += 8) {
            // Extract an 8�8 block
            float block[8][8];
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    block[i][j] = quantizedImage[row + i][col + j];
                }
            }

            // Generate RLE AC coefficients for the block
            std::vector<std::pair<int, float>> rleAC = generateRLEAC(block);
            // Write RLE AC coefficients to the file
            writeRLEToFile(rleAC, outFile);
        }
    }
    outFile.close();
    if (outFile) {
        std::cout << "RLE AC coefficients for the image successfully written to " << filename << std::endl;
    }
    else {
        std::cerr << "Error: Failed to write RLE AC coefficients to file." << std::endl;
    }
}

float** decodeRLEandDC(const char* dcFileName, const char* acFileName) {

    // Decode the delta-encoded dc file
    float** dcImage = deltaDecodeDC(dcFileName, 32);

    //  Prepare the 256x256 image for reconstruction
    float** image = new float* [256];
    for (int i = 0; i < 256; ++i) {
        image[i] = new float[256];
    }

    // Open and read the RLE AC file
    std::ifstream acFile(acFileName);

    // Decode each block
    for (int row = 0; row < 256; row += 8) {
        for (int col = 0; col < 256; col += 8) {
            // Initialize  8BY8 block with 0
            float block[8][8] = { 0 };

            // Set the DC coefficient
            block[0][0] = dcImage[row / 8][col / 8];

            // Decode RLE AC coefficients
            int zigzagIndex = 1;
            while (zigzagIndex < 64) {
                int runLength;
                float value;
                acFile >> runLength >> value;

                if (runLength == 0 && value == 0) {
                    break;
                }
                zigzagIndex += runLength;

                if (zigzagIndex >= 64) break;

                // Map the value to the appropriate zigzag position
                int r = ZIGZAG_ORDER[zigzagIndex][0];
                int c = ZIGZAG_ORDER[zigzagIndex][1];
                block[r][c] = value;

                ++zigzagIndex;
            }

            // Place the 8x8 block into the final image
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    image[row + i][col + j] = block[i][j];
                }
            }
        }
    }

    acFile.close();

    // Return the reconstructed image
    return image;
}








// ########################
// ########################
// BEGINNING OF MAIN
// ########################
// ########################

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

    cout << "##################################################################" << endl;
    cout << "####################### SESSION 01 PART 02 #######################" << endl;
    cout << "##################################################################\n" << endl;


    // Generate the cosine pattern
    float** cosinePattern = generateCosinePattern();

    // Store the cosine pattern to a RAW file
    const char* cosineFilename = "IVT2425_Abdulrahman_Almajdalawi_Session01_Part02_cosine_pattern.raw";
    store(cosineFilename, cosinePattern, 256, 256);

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

    cout << "##################################################################" << endl;
    cout << "####################### SESSION 01 PART 03 #######################" << endl;
    cout << "##################################################################\n" << endl;


    // Multiply the parrot image with the cosine pattern
    float** modifiedImage = multiplyImages(parrotImage, cosinePattern);

    // Store the modified image to a RAW file
    const char* modifiedFilename = "IVT2425_Abdulrahman_Almajdalawi_Session01_Part03_Parrot_Cosine_multiplied.raw";
    store(modifiedFilename, modifiedImage, 256, 256);


    // SESSION 2 PART 4

    cout << "##################################################################" << endl;
    cout << "####################### SESSION 02 PART 04 #######################" << endl;
    cout << "##################################################################\n" << endl;


    float** original_image = load("parrot_256x256.raw");
    float** blurred_image = load("IVT2425_Abdulrahman_Almajdalawi_Session02_Part01_blurred.raw");
    float** sharpened_image = load("IVT2425_Abdulrahman_Almajdalawi_Session02_Part01_sharpened_from_adding.raw");

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

    // SESSION 2 PART 5
    cout << "##################################################################" << endl;
    cout << "####################### SESSION 02 PART 05 #######################" << endl;
    cout << "##################################################################\n" << endl;

        // Generate uniform noise
    float** uniformNoise = generateUniformNoise(-0.5f, 0.5f);
    const char* uniformFilename = "IVT2425_Abdulrahman_Almajdalawi_Session02_Part02_uniform_noise.raw";
    store(uniformFilename, uniformNoise, 256, 256);

    // Generate Gaussian noise
    float** gaussianNoise = generateGaussianNoise(0.0f, 0.083f); // Variance equivalent to uniform noise
    const char* gaussianFilename = "IVT2425_Abdulrahman_Almajdalawi_Session02_Part02_gaussian_noise.raw";
    store(gaussianFilename, gaussianNoise, 256, 256);


    // Calculate mean and variance for uniform noise
    std::pair<float, float> uniformStats = calculateMeanAndVariance(uniformNoise);
    std::cout << "Uniform Noise - Mean: " << uniformStats.first << ", Variance: " << uniformStats.second << std::endl;

    // Calculate mean and variance for Gaussian noise
    std::pair<float, float> gaussianStats = calculateMeanAndVariance(gaussianNoise);
    std::cout << "Gaussian Noise - Mean: " << gaussianStats.first << ", Variance: " << gaussianStats.second << std::endl;

    // Session 02 Part 06
    //cout << "##################################################################" << endl;
    //cout << "####################### SESSION 02 PART 06 #######################" << endl;
    //cout << "##################################################################\n" << endl;

    float** imageWithAddedNoise = getImageWithAddedNoise(original_image, 77);
    store("IVT2425_Abdulrahman_Almajdalawi_Session02_Part06_added_noise_image.raw", imageWithAddedNoise, 256, 256);



    // Session 3 Part 07
    
    cout << "##################################################################" << endl;
    cout << "####################### SESSION 03 PART 07 #######################" << endl;
    cout << "##################################################################\n" << endl;


    float** matrixImage = createDctMatrix();
    const char* matrixfilename = "IVT2425_Abdulrahman_Almajdalawi_Session03_Part07_DCT_Matrix_256.raw";
    store(matrixfilename, matrixImage, 256, 256);

    //print2DArray(matrixImage, WIDTH, HEIGHT);
    cout << "#####" << endl;
    normalizeMatrix(matrixImage, WIDTH, HEIGHT);
    //print2DArray(matrixImage, WIDTH, HEIGHT);

    float ** transposed_matrix_image = transpose(matrixImage, 256, 256);
    float** multiplied_matrix = multiplyMatrices(matrixImage,256,256, transposed_matrix_image, 256, 256);
    store("IVT2425_Abdulrahman_Almajdalawi_Session03_Part07_Multiplied_inverse_original.raw", multiplied_matrix, 256, 256);

    // session 3 part 8
    cout << "##################################################################" << endl;
    cout << "####################### SESSION 03 PART 08 #######################" << endl;
    cout << "##################################################################\n" << endl;



    // for the parrot image signal
    // we are going to be using threshold values 10, 50, and 100 for the row extracted from the parrot image
    float* imageRow = extractRow(original_image, 10, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_random_row.raw", imageRow, 256);
    float* transformedImageRow = transformRow(imageRow, matrixImage, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_dct_random_row.raw", transformedImageRow, 256);
    float* restoredImageRow = restoreRow(transformedImageRow, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_random_row.raw", restoredImageRow, 256);
    
    //Threshold 10
    float* thresholdedtransformedImageRow01 = thresholdCoefficients(transformedImageRow,256, 10);
    float* restoredThresholdedImageRow01 = restoreRow(thresholdedtransformedImageRow01, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_random_row_thresholded_01.raw", restoredThresholdedImageRow01, 256);
    float psnrVal01 = psnrRow(imageRow, restoredThresholdedImageRow01, 256, 255);
    std::cout << "PSNR for thresholded signal with threshold val of 10 is: " << psnrVal01 << std::endl;
    
    // Threshold 50
    float* thresholdedtransformedImageRow02 = thresholdCoefficients(transformedImageRow, 256, 50);
    float* restoredThresholdedImageRow02 = restoreRow(thresholdedtransformedImageRow02, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_random_row_thresholded_02.raw", restoredThresholdedImageRow02, 256);
    float psnrVal02 = psnrRow(imageRow, restoredThresholdedImageRow02, 256, 255);
    std::cout << "PSNR for thresholded signal with threshold val of 50 is: " << psnrVal02 << std::endl;
    
    // Threshold 100
    float* thresholdedtransformedImageRow03 = thresholdCoefficients(transformedImageRow, 256, 100);
    float* restoredThresholdedImageRow03 = restoreRow(thresholdedtransformedImageRow03, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_random_row_thresholded_03.raw", restoredThresholdedImageRow03, 256);
    float psnrVal03 = psnrRow(imageRow, restoredThresholdedImageRow03, 256, 255);
    std::cout << "PSNR for thresholded signal with threshold val of 100 is: " << psnrVal03 << std::endl;




    // threshold values are going to be : 0.01, 0.5, 1
    float* noiseRow = extractRow(uniformNoise, 10, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_uniform_noise_random_row.raw", noiseRow, 256);
    float* transformedNoiseRow = transformRow(noiseRow, matrixImage, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_dct_noise_random_row.raw", transformedNoiseRow, 256);
    float* restoredNoiseRow = restoreRow(transformedNoiseRow, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_noise_random_row.raw", restoredNoiseRow, 256);

    // threshold val 1

    float* thresholdedtransformedNoiseRow01 = thresholdCoefficients(transformedNoiseRow, 256, 1);
    float* restoredThresholdedNoiseRow01 = restoreRow(thresholdedtransformedNoiseRow01, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_thresholded_noise_random_row_01.raw", restoredThresholdedNoiseRow01, 256);
    float psnrValNoise01 = psnrRow(noiseRow, restoredThresholdedNoiseRow01, 256, 1);
    std::cout << "PSNR here is for the noise row with thresholded vals of 1 : " << psnrValNoise01 << std::endl;

    // threshold val 0.5
    float* thresholdedtransformedNoiseRow02 = thresholdCoefficients(transformedNoiseRow, 256, 0.5);
    float* restoredThresholdedNoiseRow02 = restoreRow(thresholdedtransformedNoiseRow02, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_thresholded_noise_random_row_02.raw", restoredThresholdedNoiseRow02, 256);
    float psnrValNoise02 = psnrRow(noiseRow, restoredThresholdedNoiseRow02, 256, 0.5);
    std::cout << "PSNR here is for the noise row with thresholded vals of 0.5 : " << psnrValNoise02 << std::endl;

    // threshold val 0.01
    float* thresholdedtransformedNoiseRow03 = thresholdCoefficients(transformedNoiseRow, 256, 0.01);
    float* restoredThresholdedNoiseRow03 = restoreRow(thresholdedtransformedNoiseRow03, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_thresholded_noise_random_row_03.raw", restoredThresholdedNoiseRow03, 256);
    float psnrValNoise03 = psnrRow(noiseRow, restoredThresholdedNoiseRow03, 256, 0.5);
    std::cout << "PSNR here is for the noise row with thresholded vals of 0.01` : " << psnrValNoise03 << std::endl;

    //


    // Now for the cosine row pattern, we will threshold it with values of 0.01, 0.5, and 0.9
    float* cosineRow = extractRow(cosinePattern, 10, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_cosine_random_row.raw", cosineRow, 256);
    float* transformedCosineRow = transformRow(cosineRow, matrixImage, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_dct_cosine_random_row.raw", transformedCosineRow, 256);
    float* restoredCosineRow = restoreRow(transformedCosineRow, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_restored_dct_cosine_random_row.raw", restoredCosineRow, 256);
    //printRow(transformedCosineRow, 256);

    // now for the thresholding parts
    // threshold val = 0.9
    float* thresholdedtransformedCosineRow01 = thresholdCoefficients(transformedCosineRow, 256, 0.9);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_thresholdeddct_cosine_random_row.raw", thresholdedtransformedCosineRow01, 256);
    float* restoredThresholdedCosineRow01 = restoreRow(thresholdedtransformedCosineRow01, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_thresholded_cosine_random_row_01.raw", restoredThresholdedCosineRow01, 256);
    float psnrValCosine01 = psnrRow(cosineRow, restoredThresholdedCosineRow01, 256, 1);
    std::cout << "PSNR val of restored thresholded cosine signal (threshold val = 0.9) with original cosine signal here is: " << psnrValCosine01 << std::endl;
    //printRow(thresholdedtransformedCosineRow01, 256);

    // threshold val = 0.5
    float* thresholdedtransformedCosineRow02 = thresholdCoefficients(transformedCosineRow, 256, 0.5);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_thresholdeddct_cosine_random_row.raw", thresholdedtransformedCosineRow02, 256);
    float* restoredThresholdedCosineRow02 = restoreRow(thresholdedtransformedCosineRow02, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_thresholded_cosine_random_row_02.raw", restoredThresholdedCosineRow02, 256);
    float psnrValCosine02 = psnrRow(cosineRow, restoredThresholdedCosineRow02, 256, 1);
    std::cout << "PSNR val of restored thresholded cosine signal (threshold val = 0.5) with original cosine signal here is: " << psnrValCosine02 << std::endl;

    // threshold val = 0.01
    float* thresholdedtransformedCosineRow03 = thresholdCoefficients(transformedCosineRow, 256, 0.01);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_thresholdeddct_cosine_random_row.raw", thresholdedtransformedCosineRow03, 256);
    float* restoredThresholdedCosineRow03 = restoreRow(thresholdedtransformedCosineRow03, transposed_matrix_image, 256);
    storeRawRow("IVT2425_Abdulrahman_Almajdalawi_Session03_Part08_resotred_thresholded_cosine_random_row_03.raw", restoredThresholdedCosineRow03, 256);
    float psnrValCosine03 = psnrRow(cosineRow, restoredThresholdedCosineRow03, 256, 1);
    std::cout << "PSNR val of restored thresholded cosine signal (threshold val = 0.01) with original cosine signal here is: " << psnrValCosine03 << std::endl;



    // SESSION 4
    // PART 09

    cout << "##################################################################" << endl;
    cout << "####################### SESSION 04 PART 09 #######################" << endl;
    cout << "##################################################################\n" << endl;


    float** transformedImage = transform2D(parrotImage, matrixImage);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part09_dct_transformed_image.raw", transformedImage, 256, 256);
    float** restoredImage = restoreTransform2d(transformedImage, transposed_matrix_image, matrixImage);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part09_restored_transformed_image.raw", restoredImage, 256, 256);

    // now we threshold some values
    // first we try to threshold values below 10
    float** thresholdedDCT01 = threshold2D(transformedImage, 256, 10);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part09_thresholdedDCT_01_image.raw", thresholdedDCT01, 256, 256);
    float** restoredThresholded01 = restoreTransform2d(thresholdedDCT01, transposed_matrix_image, matrixImage);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part09_restored_thresholdedDCT_01_image.raw", restoredThresholded01, 256, 256);
    float psnrThresold01 = psnr(parrotImage, restoredThresholded01,255);
    cout << "The value of psnr after thresholding vals below 10 is: " << psnrThresold01 << endl;

    //  we try to threshold values below 50
    float** thresholdedDCT02 = threshold2D(transformedImage, 256, 50);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part09_thresholdedDCT_02_image.raw", thresholdedDCT02, 256, 256);
    float** restoredThresholded02 = restoreTransform2d(thresholdedDCT02, transposed_matrix_image, matrixImage);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part09_restored_thresholdedDCT_02_image.raw", restoredThresholded02, 256, 256);
    float psnrThresold02 = psnr(parrotImage, restoredThresholded02,255);
    cout << "The value of psnr after thresholding vals below 50 is: " << psnrThresold02 << endl;

    // then we try to threshold values below 100
    float** thresholdedDCT03 = threshold2D(transformedImage, 256, 100);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part09_thresholdedDCT_03_image.raw", thresholdedDCT03, 256, 256);
    float** restoredThresholded03 = restoreTransform2d(thresholdedDCT03, transposed_matrix_image, matrixImage);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part09_restored_thresholdedDCT_03_image.raw", restoredThresholded03, 256, 256);
    float psnrThresold03 = psnr(parrotImage, restoredThresholded03,255);
    cout << "The value of psnr after thresholding vals below 100 is: " << psnrThresold03 << endl;




    // Part 10
    
    cout << "##################################################################" << endl;
    cout << "####################### SESSION 04 PART 10 #######################" << endl;
    cout << "##################################################################\n" << endl;

    
    float* rows[8];
    for (int i = 0; i < 8; ++i) {
        rows[i] = QTable[i];
    };
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part10_Q_table.raw", rows, 8, 8);

    //float** dct = generate8Dct();
    //float** idct = transpose(dct, 8, 8);
    approximate(parrotImage, 256);



    // Session 04 Part 11

    cout << "##################################################################" << endl;
    cout << "####################### SESSION 04 PART 11 #######################" << endl;
    cout << "##################################################################\n" << endl;


    float** encodedImage = encode(parrotImage, 256);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part11_encodedImage.raw", encodedImage, 256, 256);
    float** decodedImage = decode(encodedImage, 256);
    store("IVT2425_Abdulrahman_Almajdalawi_Session04_Part11_decodedImage.raw", decodedImage, 256, 256);


    // Session 05 Part 12

    cout << "##################################################################" << endl;
    cout << "####################### SESSION 05 PART 12 #######################" << endl;
    cout << "##################################################################\n" << endl;


    float** downsizedImage = getDC(encodedImage);
    store("IVT2425_Abdulrahman_Almajdalawi_Session05_Part12_dcImage.raw", downsizedImage, 32, 32);
    deltaEncodeDC(downsizedImage, 32, "IVT2425_Abdulrahman_Almajdalawi_Session05_Part12_deltaEncoded.txt");
    float** imageFromDelta = deltaDecodeDC("IVT2425_Abdulrahman_Almajdalawi_Session05_Part12_deltaEncoded.txt", 32);
    store("IVT2425_Abdulrahman_Almajdalawi_Session05_Part12_restored_image_final.raw", imageFromDelta, 32, 32);


    // Session 05 Part 13
    cout << "##################################################################" << endl;
    cout << "####################### SESSION 05 PART 13 #######################" << endl;
    cout << "##################################################################\n" << endl;
    encodeRLE(encodedImage, "IVT2425_Abdulrahman_Almajdalawi_Session05_Part13_Extract_AC_RLE.txt");

    // final
    float** dctImageFinal = decodeRLEandDC("IVT2425_Abdulrahman_Almajdalawi_Session05_Part12_deltaEncoded.txt", "IVT2425_Abdulrahman_Almajdalawi_Session05_Part13_Extract_AC_RLE.txt");
    store("IVT2425_Abdulrahman_Almajdalawi_Session05_Part13_final_dct_quantized_image.raw", dctImageFinal, 256, 256);
    //now to get the restored image
    float** restoredImageFinal = decode(dctImageFinal, 256);
    store("IVT2425_Abdulrahman_Almajdalawi_Session05_Part13_final_restored_image.raw", restoredImageFinal, 256, 256);
    
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
