// See https://aka.ms/new-console-template for more information

using NeuralNetworkButReal;

Console.WriteLine("Hello, World!");
int[] sizes = new int[3] { 2, 2, 1 };
var a = new Network(sizes);

double[] inputs = new double[2] { 1.0f, 1.0f };

double[] output = a.predict(inputs);

Console.WriteLine(output[0].ToString());
