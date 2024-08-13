namespace NeuralNetworkButReal;

public class Network
{
    private int _noLayers;
    private int[] _layerSizes;
    private Layer[] _layers;

    public Network(int[] layers)
    {
        _layerSizes = layers;
        _noLayers = _layerSizes.Length;
        _layers = new Layer[_noLayers-1];
        
        for (int i = 0; i < _noLayers - 1; i++) // The last output layer does not need to exist; it does no calculation
        {
            _layers[i] = new Layer(_layerSizes[i], _layerSizes[i + 1]);
        }
    }

    public double[] predict(double[] inputs)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(inputs.Length, _layerSizes[0], nameof(inputs));
        
        Matrix<double> current = new Matrix<double>(_layerSizes[0], inputs);

        for (int i = 0; i < _noLayers - 1; i++)
        {
            current = _layers[i].feedForward(current); // feed forward through whole network
        }
        
        double[] ret = new double[current.GetHeight()];
        for (int i = 0; i < current.GetHeight(); i++)
        {
            ret[i] = current.GetValue(0, i);
        }

        return ret;
    }

    private double meanSquaredError(double[] predicted, double[] real)
    {
        if (predicted.Length != real.Length)
            throw new ArgumentException("Dimension of predicted and real vectors do not match");

        int dimension = predicted.Length;
        double total = 0;
        for (int i = 0; i < dimension; i++)
        {
            double diff = predicted[i] - real[i];
            total += diff * diff; // hence the name, mean squared error
        }

        total /= dimension; // only 1 division operation; more efficient by 0.0000001 ms 😎
        return total;
    }
    
    public void train(double[,] train, double[,] answers, int epochs = 100, double learningRate = 0.01) // optional arguments
    {
        if (train.GetLength(0) != answers.GetLength(0))
            throw new ArgumentException("Amount of training data does not match amount of answers");

        if (train.GetLength(1) != _layerSizes[0])
            throw new ArgumentException("Training data does not match dimension of first layer");

        int lastLayerSize = _layerSizes[_noLayers - 1];
        if (answers.GetLength(1) != lastLayerSize)
            throw new ArgumentException("Dimension of answers does not match dimension of last layer");

        for (int epoch = 0; epoch < epochs; epoch++) // train for the given number of epochs
        {
            for (int trainIndex = 0; trainIndex < train.GetLength(0); trainIndex++) // go over every vector in training data
            {
                double[] trainData = new double[_layerSizes[0]];
                for (int i = 0; i < _layerSizes[0]; i++)
                    trainData[i] = train[trainIndex, i]; // there is sadly no better way to do this with a uniform 2d array
                                                         // if I used double[][], an array of arrays, I could simply do
                                                         // train[trainIndex], but this does not ensure the array remains rectangular
                                                         // Thanks Microsoft :)
                                                         
                double[] predicted = predict(trainData);
                
                // we are going to use Stochastic Gradient Descent for optimisation
                
                double[] partialDerivs = new double[lastLayerSize]; // calculate partial derivatives of how predicted 
                for (int i = 0; i < lastLayerSize; i++)             // last layer values affect loss (MSE)
                {
                    // this formula comes from the definition of the mean squared error (which we are using to calculate loss)
                    // if we take the partial derivative of the formula with respect to the i'th predicted value, 
                    // d L/d Predi = 2*(Predi - Actuali) 
                    
                    partialDerivs[i] = 2 * (predicted[i] - answers[trainIndex, i]) / lastLayerSize; 
                }

                Matrix<double> p = new Matrix<double>(lastLayerSize, partialDerivs);
                // now that we have the partial derivatives, we need to pass them backwards through the network

                for (int layer = 0; layer < _noLayers - 1; layer++)
                {
                    Layer l = _layers[_noLayers - layer - 2]; // iterate backwards
                    p = l.backprop(p, learningRate); // Train all layers
                }
            }
        }

    }

}