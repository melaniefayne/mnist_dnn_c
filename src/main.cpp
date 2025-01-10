#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <sstream>
#include <string>
#include <fstream>

class NeuralNetwork
{
private:
    std::vector<int> layers; // Number of neurons in each layer

    // weights[i][j][k]: 3D vector that represents the weight connecting the k-th
    // neuron in layer i−1 to the 𝑗-th neuron in layer i.
    std::vector<std::vector<std::vector<double>>> weights;

    // biases[i][j] 2D vector that stores the j-th neuron in layer i.
    std::vector<std::vector<double>> biases;

    // 2D vector storing the output (activation) of each neuron in each layer.
    std::vector<std::vector<double>> activations;

public:
    NeuralNetwork(const std::vector<int> &layerSizes)
    {
        layers = layerSizes;
        initializeWeights();
        initializeActivations();
    }

    void initializeWeights()
    {
        srand(time(0)); // Seed for random number generation

        // Start from the second layer (i = 1) because the input layer doesn’t
        // have weights (no connections before it).
        for (size_t i = 1; i < layers.size(); ++i) // for each layer (starting from the 2nd layer)
        {
            std::vector<std::vector<double>> layerWeights(layers[i], std::vector<double>(layers[i - 1]));
            std::vector<double> layerBiases(layers[i]);

            for (size_t j = 0; j < layers[i]; ++j) // for each neuron in this layer
            {
                for (size_t k = 0; k < layers[i - 1]; ++k) // for each neuron in the previous layer
                {
                    layerWeights[j][k] = ((double)rand() / RAND_MAX) * 0.1 - 0.05; // Random weights between -0.05 & 0.05
                }
                layerBiases[j] = 0.0; // Initialize biases to 0
            }
            weights.push_back(layerWeights);
            biases.push_back(layerBiases);
        }
    }

    void initializeActivations()
    {
        for (size_t i = 0; i < layers.size(); ++i)
        {
            activations.push_back(std::vector<double>(layers[i], 0.0));
        }
    }

    std::vector<double> forwardProp(const std::vector<double> &input)
    {
        // Set input layer activations
        // ... since there was no layer before this to pass activations to the input layer
        activations[0] = input;

        for (size_t i = 1; i < layers.size(); ++i) // for each layer (starting from the 2nd layer)
        {
            for (size_t j = 0; j < layers[i]; ++j) // for each neuron in this layer
            {
                double z = biases[i - 1][j];               // Start with the bias for the previous layer
                for (size_t k = 0; k < layers[i - 1]; ++k) // for each neuron in the previous layer
                {
                    // add product of the weights and activations for the previous layer
                    z += weights[i - 1][j][k] * activations[i - 1][k];
                }

                // Apply scaled hyperbolic tangent activation
                activations[i][j] = 1.7159 * tanh(0.6666 * z);
            }
        }

        // Return the output layer activations
        return activations.back();
    }

    void backProp(const std::vector<double> &target, double learningRate)
    {
        // 2D vector that stores deltas (error gradients) for each layer
        // deltas[i][j] = how much the neuron j in layer i contributes to overall nn error
        std::vector<std::vector<double>> deltas(layers.size());

        // Compute delta for the output layer
        size_t outputLayer = layers.size() - 1;
        deltas[outputLayer].resize(layers[outputLayer]); // Ensures the output layer has space for 1 delta per neuron
        for (size_t j = 0; j < layers[outputLayer]; ++j)
        {
            double a = activations[outputLayer][j];                           // Output activation
            double error = target[j] - a;                                     // Difference between target and output
            deltas[outputLayer][j] = error * (1.7159 * (1 - a * a) / 1.7159); // Scaled tanh derivative
        }

        // Backpropagate the error to hidden layers
        for (int i = layers.size() - 2; i > 0; --i)
        {
            deltas[i].resize(layers[i]);
            for (size_t j = 0; j < layers[i]; ++j)
            {
                double z_derivative = 1.7159 * (1 - activations[i][j] * activations[i][j]) / 1.7159;
                double sum = 0.0;
                for (size_t k = 0; k < layers[i + 1]; ++k)
                {
                    sum += weights[i][k][j] * deltas[i + 1][k];
                }
                deltas[i][j] = sum * z_derivative;
            }
        }

        // Update weights and biases
        for (size_t i = 1; i < layers.size(); ++i)
        {
            for (size_t j = 0; j < layers[i]; ++j)
            {
                for (size_t k = 0; k < layers[i - 1]; ++k)
                {
                    weights[i - 1][j][k] += learningRate * deltas[i][j] * activations[i - 1][k];
                }
                biases[i - 1][j] += learningRate * deltas[i][j];
            }
        }
    }

    void train(const std::vector<std::vector<double>> &inputs,
               const std::vector<std::vector<double>> &targets,
               size_t epochs, size_t batchSize, double learningRate)
    {
        size_t numSamples = inputs.size();
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
            std::cout << "---------------" << std::endl;
            double totalLoss = 0.0;

            size_t batchCount = 0;
            for (size_t i = 0; i < numSamples; i += batchSize)
            {
                batchCount++;
                size_t currentBatchSize = std::min(batchSize, numSamples - i);
                double batchLoss = 0.0;
                size_t totalBatches = (numSamples + batchSize - 1) / batchSize;

                // Accumulate gradients over the batch
                for (size_t j = i; j < std::min(i + batchSize, numSamples); ++j)
                {

                    std::vector<double> output = forwardProp(inputs[j]);

                    // Compute loss (e.g., MSE)
                    for (size_t k = 0; k < output.size(); ++k)
                    {
                        double sampleLoss = (targets[j][k] - output[k]) * (targets[j][k] - output[k]);
                        batchLoss += sampleLoss;
                        totalLoss += sampleLoss;
                    }

                    backProp(targets[j], learningRate);
                }


                if (batchCount == 1 || batchCount % 100 == 0)
                {
                    std::cout << "Batch " << batchCount << " of " << totalBatches << ": Loss = " << batchLoss / currentBatchSize << " ..." << std::endl;
                }
            }
            std::cout << "---------------" << std::endl;
            std::cout << "Epoch Loss: " << totalLoss / numSamples << std::endl;
            std::cout << "---------------" << std::endl;
        }
    }
};

void loadMNIST(const std::string &filename,
               std::vector<std::vector<double>> &inputs,
               std::vector<std::vector<double>> &targets)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;

    // Skip the header row
    if (std::getline(file, line))
    {
        if (line.find("label") != std::string::npos)
        {
        }
    }

    // Read the data rows
    while (std::getline(file, line))
    {
        if (line.empty())
            continue; // Skip blank lines

        std::stringstream ss(line);
        std::string value;

        // Parse the label (first value in the row)
        if (!std::getline(ss, value, ','))
            continue; // Skip invalid rows
        try
        {
            int label = std::stoi(value); // Convert label to integer
            std::vector<double> target(10, 0.0);
            target[label] = 1.0;
            targets.push_back(target);
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << "Error: Invalid label in file " << filename << std::endl;
            continue; // Skip this row
        }

        // Parse the 784 pixel values
        std::vector<double> input;
        while (std::getline(ss, value, ','))
        {
            try
            {
                double pixel = std::stod(value); // Convert pixel to double
                input.push_back(pixel / 255.0);  // Normalize to [0, 1]
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Error: Invalid pixel value in file " << filename << std::endl;
                continue; // Skip invalid pixel values
            }
        }

        if (input.size() == 784)
        {
            inputs.push_back(input);
        }
        else
        {
            std::cerr << "Error: Row does not contain 784 pixel values in file " << filename << std::endl;
        }
    }

    file.close();
}

int main()
{
    std::vector<int> architecture = {784, 1000, 500, 10};
    NeuralNetwork nn(architecture);

    // Prepare input and target vectors
    std::vector<std::vector<double>> trainInputs;
    std::vector<std::vector<double>> trainTargets;
    // std::vector<std::vector<double>> testInputs;
    // std::vector<std::vector<double>> testTargets;

    // Load training and test data
    loadMNIST("data/mnist_train.csv", trainInputs, trainTargets);
    // loadMNIST("data/mnist_test.csv", testInputs, testTargets);

    int epochs = 10;
    int batchSize = 32;
    double learningRate = 0.01;
    std::cout << "Training NN" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Number of training examples: " << trainInputs.size() << std::endl;
    std::cout << "Params: epochs:" << epochs << ", batchSize:" << batchSize << " learningRate:" << learningRate << std::endl;

    // Train the network
    nn.train(trainInputs, trainTargets, 10, 32, 0.01);

    std::cout << "Training completed. Ready for testing!" << std::endl;

    return 0;
}