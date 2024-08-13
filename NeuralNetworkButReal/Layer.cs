namespace NeuralNetworkButReal;

public class Layer
{
    private Matrix<double> _weights;
    private Matrix<double> _biases;
    private int _size, _nextSize;

    private Matrix<double> _lastIn; // this variable is necessary for training
    private Matrix<double> _lastOut;
    
    public Layer(int size, int nextSize)
    {
        _size = size;
        _nextSize = nextSize;
        
        Random r = new Random();

        double[,] weightArray = new double[nextSize, size]; // size of current layer should be width
        double[,] biasArray = new double[nextSize, 1];
        
        for (int i = 0; i < nextSize; i++)
        {
            for (int j = 0; j < size; j++)
            {
                weightArray[i, j] = r.NextDouble() * 2 - 1; // Random double between -1 and 1
            }

            biasArray[i, 0] = r.NextDouble() * 2 - 1;
        }

        _weights = new Matrix<double>(size, nextSize, weightArray);
        _biases = new Matrix<double>(1, nextSize, biasArray);


        double[] temp = new double[_size];
        temp.Initialize(); // Should be all 0s
        _lastIn = new Matrix<double>(_size, temp);

        temp = new double[_nextSize];
        temp.Initialize();
        _lastOut = new Matrix<double>(_nextSize, temp);
    }

    private double sigmoid(double x) // whoa, safety by using a private method 🤯
    {
        return 1 / (1 + double.Exp(-x));
    }

    private double sigmoidDeriv(double x)
    {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    public Matrix<double> feedForward(Matrix<double> values)
    {
        _lastIn = new Matrix<double>(values); // Create a copy for storage
        Matrix<double> ret = _weights * values + _biases;

        _lastOut = new Matrix<double>(ret); // We want the output before doing the sigmoid
        
        for (int i = 0; i < _nextSize; i++)
        {
            ret.SetValue(0, i, sigmoid(ret.GetValue(0, i)));
        }
        
        return ret;
    }
    
    public Matrix<double> backprop(Matrix<double> partials, double learningRate) // we will return our partial derivatives
    {
        // why don't friend classes exist in C#
        // this method has to be public, but no user should ever be touching this method, only the Network class this is part of

        double[] derivs = new double[_size];
        derivs.Initialize(); // Set to all 0s
        
        for (int i = 0; i < _nextSize; i++)
        {
            double s = sigmoidDeriv(_lastOut.GetValue(0, i));

            for (int j = 0; j < _size; j++) 
            {
                double w = _weights.GetValue(j, i);
                
                derivs[j] += w * s * partials.GetValue(0, i); // Get derivative of node with respect loss: dL/dnodej

                // deriv of the respective weight with respect to loss
                double dL_dwji = _lastIn.GetValue(0, j) * s * partials.GetValue(0, i);
                
                // the SGD step
                // decrease the weight by the learning rate * the partial derivative to minimise loss
                _weights.SetValue(j, i, w - learningRate*dL_dwji);
            }

            // SGD of biases
            double b = _biases.GetValue(0, i);
            double dL_dbj = partials.GetValue(0, i) * s;
            _biases.SetValue(0, i, b - learningRate*dL_dbj);
        }
        // We have now updated all the weights and biases, and calculated the partial derivatives of the inputs
        Matrix<double> ret = new Matrix<double>(_size, derivs);
        return ret;
    }
}