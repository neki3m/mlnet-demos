using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;


namespace ConsolaML
{
    public static class RealTaxiFareModel
    {
        // \bin\Debug\netcoreapp2.0\Data
        const string DataPath = @"..\..\..\Data\yellow_tripdata_2017-02.csv";
        const string TestDataPath = @"..\..\..\Data\yellow_tripdata_2017-01.csv";
        const string ModelPath = @"..\..\..\Models\Model.zip";


        public static void CalcularModelo()
        {
            Console.WriteLine(DateTime.Now.ToLongTimeString());
            var pipeline = new LearningPipeline();

            //carga los datos

            pipeline.Add(new TextLoader<RealTaxiTrip>(DataPath, useHeader: true, separator: ","));

            //copia los datos de la columna fare_amount a na nueva columna llamada label
            pipeline.Add(new ColumnCopier(("fare_amount", "Label")));

            //pasa el texo a numeros para poder tratarlos por el algoritomo.Como estas features son categoricas
            //crea un vector de valores para cada una de ellas. Cambia cada caegoria por un numero en cada una de esas columnas
            pipeline.Add(new CategoricalOneHotVectorizer("VendorID", "RatecodeID", "payment_type"));

            //junta todo en una columan llamadas Features para que sea más facil de tratart por el algoritmo
            //solo hace falta meter las columnas que crees que influyen, no tienes pq meter todas
            pipeline.Add(new ColumnConcatenator("Features", "VendorID", "RatecodeID", "passenger_count", "trip_distance", "payment_type"));

            //elegimos el algoritmo que queremos usar para entrenar el modelo
            pipeline.Add(new FastTreeRegressor());

            Console.WriteLine("Pulse una tecla para entrena");
            Console.ReadLine();
            Console.WriteLine("A entrenar! " + DateTime.Now.ToLongTimeString());
            //vamos a entrenar el modelo
            PredictionModel<RealTaxiTrip, RealTaxiTripFarePrediction> model = pipeline.Train<RealTaxiTrip, RealTaxiTripFarePrediction>();
            //await model.WriteAsync(ModelPath);

            Console.WriteLine("Entrenado " + DateTime.Now.ToLongTimeString());
            //evaluamos el modelo
            //cargamos los datos de pruebas
            var testData = new TextLoader<RealTaxiTrip>(TestDataPath, useHeader: true, separator: ",");

            Console.WriteLine("Probado " + DateTime.Now.ToLongTimeString());


            //evaluamos con la metrica de regresion. EN funcion de los algoritmos hay evaluadores mejores y peores
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            // Rms should be around 2.795276 -  The lower it is, the better your model.
            Console.WriteLine("Rms=" + metrics.Rms + " The lower it is, the better your model");

            //otro evaluador RSquared will be a value between 0 and 1. The closer you are to 1, the better your model.
            Console.WriteLine("RSquared = " + metrics.RSquared + " The closer you are to 1, the better your model.");

            Console.WriteLine("Evaluado " + DateTime.Now.ToLongTimeString());



            //predecir un valor

            RealTaxiTrip Trip1 = new RealTaxiTrip
            {
                VendorID = "VTS",
                RatecodeID = "1",
                passenger_count = 1,
                trip_distance = 10.33f,
                payment_type = "CSH",
                fare_amount = 0 // predict it. actual = 29.5
            };

            var prediction = model.Predict(Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.fare_amount);

            Console.WriteLine("Predecido " + DateTime.Now.ToLongTimeString());
        }
    }
    public class RealTaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float fare_amount;
    }

    public class RealTaxiTrip
    {
        [Column(ordinal: "0")]
        public string VendorID;

        [Column(ordinal: "5")]
        public string RatecodeID;

        [Column(ordinal: "3")]
        public float passenger_count;

        //[Column(ordinal: "3")]
        //public float trip_time_in_secs;

        [Column(ordinal: "4")]
        public float trip_distance;

        [Column(ordinal: "9")]
        public string payment_type;

        [Column(ordinal: "10")]
        public float fare_amount;
    }


}
