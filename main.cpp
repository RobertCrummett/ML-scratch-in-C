#include "mkl.h"
#include <iostream>
#include <random>

typedef struct DenseLayer {
        int InputNumber;
        int NeuronNumber;
        double* Weights;
        double* Biases;
} DenseLayer;

static double* AllocateMatrix(int Rows, int Cols) {
        double* Matrix = new double[Rows * Cols];
        if (Matrix == nullptr) {
                std::cerr << "Memory allocation failed" << std::endl;
                std::exit(EXIT_FAILURE);
        }

        std::mt19937 Engine(std::random_device{}());
        std::uniform_real_distribution<double> Dist(0.0, 1.0); // Range [0, 1)

        for (int i = 0; i < Rows * Cols; ++i)
                Matrix[i] = Dist(Engine);

        return Matrix;
}

static DenseLayer* AllocateDenseLayer(int InputNumber, int NeuronNumber) {
        DenseLayer* Layer = new DenseLayer;
        if (Layer == nullptr) {
                std::cerr << "Memory allocation failed" << std::endl;
                std::exit(EXIT_FAILURE);
        }
        Layer->InputNumber = InputNumber;
        Layer->NeuronNumber = NeuronNumber;
        Layer->Weights = AllocateMatrix(InputNumber, NeuronNumber);
        Layer->Biases = AllocateMatrix(1, NeuronNumber);
        return Layer;
}

static void FreeDenseLayer(DenseLayer* Layer) {
        if (Layer != nullptr) {
                if (Layer->Weights != nullptr)
                        delete[] Layer->Weights;
                if (Layer->Biases != nullptr)
                        delete[] Layer->Biases;
                delete[] Layer;
        }
}

static void PrintMatrix(const double* Matrix, int Rows, int Cols) {
        for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                        std::cout << Matrix[i * Cols + j] << " ";
        std::cout << std::endl;
}

static double* Forward(DenseLayer* Layer, double* Matrix, int BatchNumber, double* Output) {
        double Alpha = 1.0;
        double Beta = 0.0;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                Layer->InputNumber, BatchNumber, Layer->NeuronNumber,
                Alpha, Layer->Weights, Layer->NeuronNumber,
                Matrix, BatchNumber, Beta, Output, BatchNumber);
}

int main() {
        int InputNumber = 2;
        int NeuronNumber = 8;
        int OutputNumber = 2;

        DenseLayer* Layer1 = AllocateDenseLayer(InputNumber, NeuronNumber);
        DenseLayer* Layer2 = AllocateDenseLayer(NeuronNumber, NeuronNumber);
        DenseLayer* Layer3 = AllocateDenseLayer(NeuronNumber, OutputNumber);

        std::cout << "Weights (Layer1):" << std::endl;
        PrintMatrix(Layer1->Weights, Layer1->InputNumber, Layer1->NeuronNumber);
        std::cout << "Biases (Layer1):" << std::endl;
        PrintMatrix(Layer1->Biases, 1, Layer1->NeuronNumber);

        FreeDenseLayer(Layer1);
        FreeDenseLayer(Layer2);
        FreeDenseLayer(Layer3);
        
        return 0;
}