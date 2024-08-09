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
        _layers = new Layer[_noLayers];
        
        for (int i = 0; i < _noLayers - 2; i++) // The last output layer does not need to exist; it does no calculation
        {
            _layers[i] = new Layer(_layerSizes[i], _layerSizes[i + 1]);
        }
    }

    public double[] predict(double[] inputs)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(inputs.Length, _layerSizes[0], nameof(inputs));
        
        Matrix<double> current = new Matrix<double>(_layerSizes[0], inputs);

        for (int i = 0; i < _noLayers - 2; i++)
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

}