/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make hough
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <vector>
#include <utility>
#include "common/pgm.h"


const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

void calculateSDyMEAN(int data[], int n, float *mean, float *stddev) {
  float sum = 0.0;
  int i;

  for(i = 0; i < n; ++i) {
    sum += data[i];
  }

  *mean = sum / n;

  sum = 0.0;

  for(i = 0; i < n; ++i) {
    sum += pow(data[i] - *mean, 2);
  }

  *stddev = sqrt(sum / n);
}
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************


// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;


  // gloID es el id global del thread y cada thread representa un pixel. El id empieza en la esquina superior izquierda
  //  y sigue a la derecha en la línea y luego sigue con la línea de abajo.
  //
  // xCoord: se hace módulo con el ancho de la imágen para encontrar el pixel en x y luego se le resta xCent para
  //  tener 0 en el centro, negativos a la izquierda y positivos a la derecha.
  // yCoord: se divide el gloId por el ancho de la imágen para obtener el número de línea (posición en y) luego se
  //  le restaría el yCent para hacer lo mismo que en x pero como la coordenada es al reves se "multiplica por -1"
  //  entonces termina siendo yCent - gloId / w
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique

          // aunque es un thread por pixel el acumulador no está definido por pixeles sino que varios
          // threads pueden accederlo por el sistema de votación, por esto se necesita atomic para evitar
          // el race condition
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (argc <= 1) {
    printf("Please pass the filename of the image as an argument\n");
    return -1;
  }

  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);
  cudaEventRecord(start);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  cudaEventRecord(stop);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  bool ranSuccessfully = true;
  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (abs(cpuht[i] - h_hough[i]) > 1){
      ranSuccessfully = false;
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
  }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  if (ranSuccessfully) {
    printf("Calculated Hough Transform in %f ms\n", milliseconds);
    float mean, stddev;
    calculateSDyMEAN(h_hough, degreeBins * rBins, &mean, &stddev);
    std::vector<std::pair<int, int>> lines;
    for (i = 0; i < degreeBins * rBins; i++){
      if (h_hough[i] > (mean + 5 * stddev)) {
        // pair order: r, th
        int my_r = i / degreeBins;
        int my_th = i % degreeBins;
        std::pair<int, int> line = {my_r, my_th};
        lines.push_back(line);
      }
    }
    inImg.writeJPEGWithLines("output.jpg", lines, radInc, rBins);
  } else
    printf("There was a problem in the calculations :(\n");

  // clean-up
  inImg.~PGMImage();
  cudaFree((void *) d_Cos);
  cudaFree((void *) d_Sin);
  cudaFree((void *) d_in);
  cudaFree((void *) d_hough);
  delete[]pcCos;
  delete[]pcSin;
  delete[]h_hough;
  cudaDeviceReset();

  return 0;
}
