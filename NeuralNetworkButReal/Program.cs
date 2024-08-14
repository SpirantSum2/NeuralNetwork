// See https://aka.ms/new-console-template for more information

using NeuralNetworkButReal;

Console.WriteLine("Hello, World!");
int[] sizes = new int[3] { 2, 3, 1 };
var a = new Network(sizes);

double[] inputs = new double[2] { 1.0f, 1.0f };

double[] output = a.predict(inputs);

Console.WriteLine(output[0].ToString());

// the xor function as training data
double[,] trainData = new double[4, 2] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
double[,] ans = new double[4, 1] { { 0 }, { 1 }, { 1 }, { 0 } };

a.train(trainData, ans, epochs:10000, learningRate:0.01);

output = a.predict(inputs);
Console.WriteLine(output[0].ToString());
