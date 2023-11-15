/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include <cmath>
#include <jpeglib.h>
#include <png.h>
#include <iostream>
#include <setjmp.h>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
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
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];


// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  //TODO calcular: int gloID = ?
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
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
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

//*****************************************************************

void drawLine(unsigned char *img, int w, int h, float rScale, int x0, int y0, int x1, int y1, unsigned char color)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2; /* error value e_xy */

    while (true)
    {
        if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
        {
            img[y0 * w + x0] = color; // Establecer el color del píxel
        }
        if (x0 == x1 && y0 == y1)
            break;
        e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        } 
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        } 
    }
}



void drawLines(unsigned char *img, int w, int h, int *acc, int threshold, float rScale)
{
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
        float theta = tIdx * radInc;
        for (int rIdx = 0; rIdx < rBins; rIdx++)
        {
            int idx = rIdx * degreeBins + tIdx;
            if (acc[idx] > threshold)
            {
                float r = (rIdx - rBins / 2) * rScale;  // Ajustar el valor de r

                // Calcular las coordenadas de la línea
                int x0, y0, x1, y1;
                if (fabs(theta) < 1e-3 || fabs(theta - M_PI) < 1e-3) // aproximadamente vertical
                {
                    x0 = x1 = static_cast<int>(r + w / 2);
                    y0 = 0;
                    y1 = h - 1;
                }
                else if (fabs(theta - M_PI / 2) < 1e-3 || fabs(theta - 3 * M_PI / 2) < 1e-3) // aproximadamente horizontal
                {
                    y0 = y1 = static_cast<int>(r + h / 2);
                    x0 = 0;
                    x1 = w - 1;
                }
                else
                {
                    // Calcular las coordenadas de la línea de manera general
                    x0 = static_cast<int>(w / 2 + r / cos(theta));
                    y0 = static_cast<int>(h / 2 - r / sin(theta));
                    x1 = static_cast<int>(w / 2 + (r - h) / cos(theta));
                    y1 = static_cast<int>(h / 2 - (r - h) / sin(theta));
                }

                // Dibujar la línea en la imagen
                drawLine(img, w, h, rScale, x0, y0, x1, y1, 255);
            }
        }
    }
}


int main(int argc, char **argv) {
  for (int run = 0; run < 10; ++run) { // Add a loop to run the main content 10 times
            printf("Execution number: %d\n", run + 1); // Display the current execution number (starts from 1)
    int i;
    PGMImage inImg(argv[1]);
    int w = inImg.x_dim;
    int h = inImg.y_dim;
    int *cpuht;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Cálculo en la CPU
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    float pcCos[degreeBins], pcSin[degreeBins];
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    // Configuración y ejecución del kernel
    unsigned char *d_in;
    int *d_hough;
    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil((w * h) / 256.0);
    cudaEventRecord(start);
    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    // Registro del final del evento
    cudaEventRecord(stop);

    // Esperar a que termine el kernel
    cudaEventSynchronize(stop);

    // Calculo del tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución del kernel: %f ms\n", milliseconds);
    printf("Done!\n");


    // Destrucción de eventos CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copiar resultados de vuelta al host y limpiar
    int *h_hough = (int *)malloc(sizeof(int) * degreeBins * rBins);
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    
    // compare CPU and GPU results
    for (i = 0; i < degreeBins * rBins; i++)
    {
      if (cpuht[i] != h_hough[i])
        printf(" ");
    }

   

    // Dibujar las líneas en la imagen
    int threshold = 3000; 
    drawLines(inImg.pixels, w, h, h_hough, threshold, rScale);

    // Guardar la imagen resultante en formato JPEG
    FILE *outfile = fopen("output_constante.jpg", "wb");
    if (!outfile) {
        std::cerr << "No se pudo abrir output.jpg para escritura" << std::endl;
        return -1;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer;
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = (JSAMPROW) &inImg.pixels[cinfo.next_scanline * w];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);

    free(cpuht);
    free(h_hough);
    cudaFree(d_in);
    cudaFree(d_hough);

  }

    return 0;
}