using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowSharpInceptionv3Scoring
{
    class Program
    {
        static string dnnModelsFolder, dnnModelFile, labelsFile;
        static string imagesFolder;

        static void Main(string[] args)
        {
            dnnModelsFolder = "DNNModels";
            imagesFolder = "ImagesForInference";

            List<string> imageFileNamesToInfer = new List<string>() { 
                                                                      "Jersey-Red.jpg",
                                                                      "mug-white.jpg",
                                                                      "t-shirt-dotnet-4.0.jpg",
                                                                      "green-frisbee.jpg" //This image should not be identified with Inception v3 with current training/config
                                                                    };

            // Run inference on the image files
            foreach (var fileName in imageFileNamesToInfer)
            {
                IEnumerable<string> tagsForCurrentImage = null;

                var filePathName = imagesFolder + "/" + fileName;

                var imageBytes = File.ReadAllBytes(filePathName);

                if (!imageBytes.IsValidImage())
                {
                    Console.WriteLine($"Error: UnsupportedMediaType");
                    Environment.Exit(1);
                }

                TensorFlowInceptionPrediction tensorFlowInceptionv3Model = new TensorFlowInceptionPrediction();

                // Calling an async method from Main()
                //
                try
                {
                    // Start a task - calling an async function

                    //-------------------------------------
                    // Calling Asyc Function from Main()
                    //
                    Task<IEnumerable<string>> callTask = Task.Run(() => tensorFlowInceptionv3Model.ClassifyImageEnumAsync(imageBytes));
                    
                    // Wait for it to finish
                    callTask.Wait();

                    // Get the result
                    tagsForCurrentImage = callTask.Result;

                    //-------------------------------------

                    // If calling from async method instead of Main()
                    //tagsForCurrentImage = await tensorFlowInceptionv3Model.ClassifyImageEnumAsync(imageBytes));


                    Console.WriteLine("=====================================================================");
                    Console.WriteLine(" ");

                    Console.WriteLine($"Image file{fileName} is classified by TensorFlow model as the following tags: ");

                    foreach (var tag in tagsForCurrentImage)
                    {
                        Console.WriteLine($"Tag: {tag}");
                    }

                    Console.WriteLine(" ");
                    Console.WriteLine("=====================================================================");


                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Caught Exception: {ex.Message}");
                }
            }

            Console.WriteLine(" ");
            Console.WriteLine("======================= END OF PROCESS ========================");

        }
    }

    public static class ImageValidationExtensions
    {
        public static bool IsValidImage(this byte[] image)
        {
            var imageFormat = GetImageFormat(image);
            return imageFormat == ImageFormat.jpeg ||
                   imageFormat == ImageFormat.png;
        }

        public enum ImageFormat
        {
            bmp,
            jpeg,
            gif,
            tiff,
            png,
            unknown
        }

        private static ImageFormat GetImageFormat(byte[] bytes)
        {
            // see http://www.mikekunz.com/image_file_header.html  
            var bmp = Encoding.ASCII.GetBytes("BM");     // BMP
            var gif = Encoding.ASCII.GetBytes("GIF");    // GIF
            var png = new byte[] { 137, 80, 78, 71 };    // PNG
            var tiff = new byte[] { 73, 73, 42 };         // TIFF
            var tiff2 = new byte[] { 77, 77, 42 };         // TIFF
            var jpeg = new byte[] { 255, 216, 255, 224 }; // jpeg
            var jpeg2 = new byte[] { 255, 216, 255, 225 }; // jpeg canon

            if (bmp.SequenceEqual(bytes.Take(bmp.Length)))
                return ImageFormat.bmp;

            if (gif.SequenceEqual(bytes.Take(gif.Length)))
                return ImageFormat.gif;

            if (png.SequenceEqual(bytes.Take(png.Length)))
                return ImageFormat.png;

            if (tiff.SequenceEqual(bytes.Take(tiff.Length)))
                return ImageFormat.tiff;

            if (tiff2.SequenceEqual(bytes.Take(tiff2.Length)))
                return ImageFormat.tiff;

            if (jpeg.SequenceEqual(bytes.Take(jpeg.Length)))
                return ImageFormat.jpeg;

            if (jpeg2.SequenceEqual(bytes.Take(jpeg2.Length)))
                return ImageFormat.jpeg;

            return ImageFormat.unknown;
        }
    }

}
