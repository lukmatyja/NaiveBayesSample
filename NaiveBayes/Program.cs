    using System;

namespace NaiveBayes
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Begin naive Bayes demo");

            Console.WriteLine("Data looks like: ");
            Console.WriteLine("Height Weight Foot Sex");
            Console.WriteLine("======================");
            Console.WriteLine("6.00,  180,  11,  0");
            Console.WriteLine("5.30,  120,   7,  1");
            Console.WriteLine(" . . .");

            double[][] data = new double[8][];
            data[0] = new double[] { 6.00, 180, 11, 0 };
            data[1] = new double[] { 5.90, 190, 9, 0 };
            data[2] = new double[] { 5.70, 170, 8, 0 };
            data[3] = new double[] { 5.60, 140, 10, 0 };

            data[4] = new double[] { 5.80, 120, 9, 1 };
            data[5] = new double[] { 5.50, 150, 6, 1 };
            data[6] = new double[] { 5.30, 120, 7, 1 };
            data[7] = new double[] { 5.00, 100, 5, 1 };

            int N = data.Length;  // 8 items

            // compute class counts

            int[] classCts = new int[2];  // male, female
            for (int i = 0; i < N; ++i)
            {
                int c = (int)data[i][3];  // class is at [3]
                ++classCts[c];
            }

            // compute and display means

            double[][] means = new double[2][];
            for (int c = 0; c < 2; ++c)
                means[c] = new double[3];

            for (int i = 0; i < N; ++i)
            {
                int c = (int)data[i][3];
                for (int j = 0; j < 3; ++j)  // ht, wt, foot
                    means[c][j] += data[i][j];
            }

            for (int c = 0; c < 2; ++c)
            {
                for (int j = 0; j < 3; ++j)
                    means[c][j] /= classCts[c];
            }

            Console.WriteLine("Means of height, weight, foot:");
            for (int c = 0; c < 2; ++c)
            {
                Console.Write("class: " + c + "  ");
                for (int j = 0; j < 3; ++j)
                    Console.Write(means[c][j].
                      ToString("F2").PadLeft(8) + " ");
                Console.WriteLine("");
            }

            // compute and display variances

            double[][] variances = new double[2][];
            for (int c = 0; c < 2; ++c)
                variances[c] = new double[3];

            for (int i = 0; i < N; ++i)
            {
                int c = (int)data[i][3];
                for (int j = 0; j < 3; ++j)
                {
                    double x = data[i][j];
                    double u = means[c][j];
                    variances[c][j] += (x - u) * (x - u);
                }
            }

            for (int c = 0; c < 2; ++c)
            {
                for (int j = 0; j < 3; ++j)
                    variances[c][j] /= classCts[c] - 1;
            }

            Console.WriteLine("Variances of ht, wt, foot:");
            for (int c = 0; c < 2; ++c)
            {
                Console.Write("class: " + c + "  ");
                for (int j = 0; j < 3; ++j)
                    Console.Write(variances[c][j].
                      ToString("F6").PadLeft(12));
                Console.WriteLine("");
            }

            // set up item to predict

            double[] unk = new double[] { 5.60, 150, 8 };
            Console.WriteLine("Item to predict:");
            Console.WriteLine("5.60   150   8");

            // compute and display conditional probs

            double[][] condProbs = new double[2][];
            for (int c = 0; c < 2; ++c)
                condProbs[c] = new double[3];

            for (int c = 0; c < 2; ++c)  // each class
            {
                for (int j = 0; j < 3; ++j)  // each predictor
                {
                    double u = means[c][j];
                    double v = variances[c][j];
                    double x = unk[j];
                    condProbs[c][j] = ProbDensFunc(u, v, x);
                }
            }


            for (int c = 0; c < 2; ++c)  // each class
            {
                for (int j = 0; j < 3; ++j)  // each predictor
                {
                    double u = means[c][j];
                    double v = variances[c][j];
                    double x = unk[j];
                    condProbs[c][j] = ProbDensFunc(u, v, x);
                }
            }

            double[] classProbs = new double[2];
            for (int c = 0; c < 2; ++c)
                classProbs[c] = (classCts[c] * 1.0) / N;

            Console.WriteLine("Unconditional probs:");
            for (int c = 0; c < 2; ++c)
                Console.WriteLine("class: " + c + "   " +
                  classProbs[c].ToString("F4"));
            // compute and display unconditional probs
            // compute and display evidence terms
            // compute and display prediction probs

            Console.WriteLine("End demo");
            Console.ReadLine();
        }

        static double ProbDensFunc(double u, double v,
      double x)
        {
            double left =
              1.0 / Math.Sqrt(2 * Math.PI * v);
            double right =
              Math.Exp(-(x - u) * (x - u) / (2 * v));
            return left * right;
        }
    }
}
