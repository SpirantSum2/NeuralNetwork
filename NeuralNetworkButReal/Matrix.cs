using System;
using System.Numerics;

namespace NeuralNetworkButReal
{
    public class Matrix<T> where T: INumber<T> // template class because I hate myself
                           // There isn't INumber until .NET 7????? How did people cope until 2023
    {
        private int _width, _height; 
        private T[,] _vals; // this type of 2d array guarantees that it remains a rectangle, unlike the T[][] jagged array 
        public Matrix(int w, int h, T[,] values) // constructor
        {
            if (w <= 0)
                throw new ArgumentException("Matrix width must be a positive integer", nameof(w));
            if (h <= 0)
                throw new ArgumentException("Matrix height must be a positive integer", nameof(h));
            
            _width = w;
            _height = h;

            if (values.GetLength(0) != _height)
                throw new ArgumentException("Matrix values height does not match argument", nameof(values));
            if (values.GetLength(1) != _width)
                throw new ArgumentException("Matrix values height does not match argument", nameof(values));

            _vals = values;
        }

        public Matrix(int h, T[] values) // a vector
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(h, nameof(h));

            _width = 1;
            _height = h;

            if (values.GetLength(0) != _height)
                throw new ArgumentException("Vector values length does not match argument");

            _vals = new T[_height, _width];
            
            for (int i = 0; i < _height; i++)
            {
                _vals[i, 0] = values[i];
            }
        }

        public Matrix(Matrix<T> copyMatrix) // method for creating a copy of a Matrix
        {
            _width = copyMatrix.GetWidth();
            _height = copyMatrix.GetHeight();

            _vals = new T[_height,_width];
            
            for (int i = 0; i < _width; i++)
            {
                for (int j = 0; j < _height; j++)
                {
                    _vals[j, i] = copyMatrix.GetValue(i, j);
                }
            }
        }

        public int GetWidth() 
        {
            return _width;
        }
        
        public int GetHeight()
        {
            return _height;
        }

        public T GetValue(int row, int column)
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, _height);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, _width);
            
            return _vals[column, row];
        }

        public void SetValue(int row, int column, T newVal)
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, _height);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, _width);

            _vals[column, row] = newVal;
        }
        
        public static Matrix<T> operator *(Matrix<T> a, Matrix<T> b) // There's no way to be able to add a Matrix<float> and a
        {                                                            // Matrix <int> or similar, so both have to be same type
                                                                     // It's the user's problem to figure this stuff out
            int dotSize = a.GetWidth();
            if (dotSize != b.GetHeight())
                throw new ArgumentException("First matrix width must equal second matrix height");

            int newWidth = b.GetWidth();
            int newHeight = a.GetHeight();
            
            T[,] newVals = new T[newHeight, newWidth]; // Pray that dynamic allocation in the heap works
                                                       // C++ sucks but at least I know what the computer is doing

           for (int i = 0; i < newHeight; i++)
           {
               for (int j = 0; j < newWidth; j++)
               {
                   T total = T.Zero; // The zero value for all classes implement the INumber interface
                   
                   for (int k = 0; k < dotSize; k++)
                   {
                       total += a.GetValue(k, i) * b.GetValue(j, k);
                   }

                   newVals[i, j] = total;
               }
           }
            
            return new Matrix<T>(newWidth, newHeight, newVals);
        }

        public static Matrix<T> operator +(Matrix<T> a, Matrix<T> b)
        {
            if (a.GetWidth() != b.GetWidth())
                throw new ArgumentException("Matrices must have the same width to be added");
            if (a.GetHeight() != b.GetHeight())
                throw new ArgumentException("Matrices must have the same height to be added");

            T[,] newVals = new T[a.GetHeight(), a.GetWidth()];
            
            for (int i = 0; i < a.GetHeight(); i++)
            {
                for (int j = 0; j < a.GetWidth(); j++)
                {
                    newVals[i, j] = a.GetValue(j, i) + b.GetValue(j, i);
                }
            }

            return new Matrix<T>(a.GetWidth(), a.GetHeight(), newVals);
        }
        
    }
}