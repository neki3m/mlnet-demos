using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace ConsolaML
{
    class Program
    {
       

        static void Main(string[] args)
        {
            // IrisFlowerModel.CalculateModel();
            TaxiFareModel.CalcularModelo();
            //RealTaxiFareModel.CalcularModelo();
           // SentimentModel.CalcularModelo();

            Console.ReadLine();
        }
    }
}
