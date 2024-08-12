namespace NeuralNetworkButReal;

public class Layer
{
    private Matrix<double> _weights;
    private Matrix<double> _biases;
    private int _size, _nextSize;
    
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
    }

    private double sigmoid(double x)
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
        Matrix<double> ret = _weights * values + _biases;

        for (int i = 0; i < _nextSize; i++)
        {
            ret.SetValue(0, i, sigmoid(ret.GetValue(0, i)));
        }

        return ret;
    }
}