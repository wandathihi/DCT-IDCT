#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

const double PI = 3.14159265358979323846;

void dctTransform(const cv::Mat& image, cv::Mat& dctImage)
{
    int rows = image.rows;
    int cols = image.cols;

    float ci, cj, dct1, sum;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ci = (i == 0) ? 1.0 / sqrt(rows) : sqrt(2.0) / sqrt(rows);
            cj = (j == 0) ? 1.0 / sqrt(cols) : sqrt(2.0) / sqrt(cols);

            sum = 0;
            for (int k = 0; k < rows; k++) {
                for (int l = 0; l < cols; l++) {
                    dct1 = image.at<float>(k, l) *
                        cos((2 * k + 1) * i * PI / (2 * rows)) *
                        cos((2 * l + 1) * j * PI / (2 * cols));
                    sum = sum + dct1;
                }
            }
            dctImage.at<float>(i, j) = ci * cj * sum;
        }
    }
}

void inverseDCTTransform(const cv::Mat& dctImage, cv::Mat& inverseImage)
{
    int rows = dctImage.rows;
    int cols = dctImage.cols;

    float ci, cj, idct1, sum;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum = 0;
            for (int k = 0; k < rows; k++) {
                for (int l = 0; l < cols; l++) {
                    ci = (k == 0) ? 1.0 / sqrt(rows) : sqrt(2.0) / sqrt(rows);
                    cj = (l == 0) ? 1.0 / sqrt(cols) : sqrt(2.0) / sqrt(cols);

                    idct1 = dctImage.at<float>(k, l) *
                        cos((2 * i + 1) * k * PI / (2 * rows)) *
                        cos((2 * j + 1) * l * PI / (2 * cols));
                    sum = sum + ci * cj * idct1;
                }
            }
            inverseImage.at<float>(i, j) = sum;
        }
    }
}

int main() {
    cv::Mat image = cv::imread("Lena.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat floatImage = image.clone();  // Clone the image to a float format
    floatImage.convertTo(floatImage, CV_32F);  // Convert to float

    int rows = floatImage.rows;
    int cols = floatImage.cols;

    int blockRows = rows / 8;
    int blockCols = cols / 8;

    cv::Mat transformedImage = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat inverseTransformedImage = cv::Mat::zeros(rows, cols, CV_32F);

    // DCT transform on each block
    for (int i = 0; i < blockRows; i++) {
        for (int j = 0; j < blockCols; j++) {
            cv::Mat block = floatImage(cv::Rect(j * 8, i * 8, 8, 8));
            cv::Mat dctBlock = transformedImage(cv::Rect(j * 8, i * 8, 8, 8));
            dctTransform(block, dctBlock);
        }
    }

    //Inverse DCT transform on each block
    for (int i = 0; i < blockRows; i++) {
        for (int j = 0; j < blockCols; j++) {
            cv::Mat dctBlock = transformedImage(cv::Rect(j * 8, i * 8, 8, 8));
            cv::Mat inverseBlock = inverseTransformedImage(cv::Rect(j * 8, i * 8, 8, 8));
            inverseDCTTransform(dctBlock, inverseBlock);
        }
    }
    //Convert transformed images to displayable foemat
    cv::Mat displayTransformedImage;
    transformedImage.convertTo(displayTransformedImage, CV_8U);

    cv::Mat displayInverseImage;
    inverseTransformedImage.convertTo(displayInverseImage, CV_8U);
    //show and save image
    cv::imshow("Original Image", image);
    cv::imshow("Transformed Image (DCT Coefficients)", displayTransformedImage);
    cv::imshow("Inverse Transformed Image", displayInverseImage);
    cv::imwrite("transformed_dct.jpg", displayTransformedImage);
    cv::imwrite("inverse_transformed_idct.jpg", displayInverseImage);

    cv::waitKey(0);

    return 0;
}

