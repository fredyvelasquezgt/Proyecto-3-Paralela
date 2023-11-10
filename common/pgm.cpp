/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Used in different projects to handle PGM I/O
 To build use  : 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <jpeglib.h>
#include <math.h>
#include <vector>
#include "pgm.h"

using namespace std;

//-------------------------------------------------------------------
PGMImage::PGMImage(char *fname)
{
   x_dim=y_dim=num_colors=0;
   pixels=NULL;
   
   FILE *ifile;
   ifile = fopen(fname, "rb");
   if(!ifile) return;

   char *buff = NULL;
   size_t temp;

   fscanf(ifile, "%*s %i %i %i", &x_dim, &y_dim, &num_colors);

   getline((char **)&buff, &temp, ifile); // eliminate CR-LF
   
   assert(x_dim >1 && y_dim >1 && num_colors >1);
   pixels = new unsigned char[x_dim * y_dim];
   fread((void *) pixels, 1, x_dim*y_dim, ifile);   
   
   fclose(ifile);
}
//-------------------------------------------------------------------
PGMImage::PGMImage(int x=100, int y=100, int col=16)
{
   num_colors = (col>1) ? col : 16;
   x_dim = (x>1) ? x : 100;
   y_dim = (y>1) ? y : 100;
   pixels = new unsigned char[x_dim * y_dim];
   memset(pixels, 0, x_dim * y_dim);
   assert(pixels);
}
//-------------------------------------------------------------------
PGMImage::~PGMImage()
{
  if(pixels != NULL)
     delete [] pixels;
  pixels = NULL;
}
//-------------------------------------------------------------------
bool PGMImage::write(char *fname)
{
   int i,j;
   FILE *ofile;
   ofile = fopen(fname, "w+t");
   if(!ofile) return 0;

   fprintf(ofile,"P5\n%i %i\n%i\n",x_dim, y_dim, num_colors);
   fwrite(pixels, 1, x_dim*y_dim, ofile);
   fclose(ofile);
   return 1;
}

void PGMImage::writeJPEGWithLines(const char* filename, std::vector<std::pair<int, int>> lines, float radInc, int rBins)
{
   struct jpeg_compress_struct cinfo;
   struct jpeg_error_mgr jerr;
   unsigned char *new_pixels = new unsigned char[x_dim * y_dim * 3];
   float rMax = sqrt (1.0 * x_dim * x_dim + 1.0 * y_dim * y_dim) / 2;
   float rScale = 2 * rMax / rBins;
   int xCent = x_dim / 2;
   int yCent = y_dim / 2;

   for (int i = 0; i < x_dim; i++) {
      for (int j = 0; j < y_dim; j++) {
         bool isLine = false;
         int xCoord = i - xCent;
         int yCoord = yCent - j;

         for (std::pair<int, int> line: lines) {
            int rIdx = line.first;
            int thIdx = line.second;
            float r = rIdx * rScale - rMax;
            float th = thIdx * radInc;
            // r = x*cos(th) + y*sin(th)
            if (abs(r - xCoord * cos(th) - yCoord * sin(th)) < 0.5) {
               isLine = true;
               break;
            }
         }


         if (isLine) {
            new_pixels[(j*x_dim + i)*3] = 255;
            new_pixels[(j*x_dim + i)*3+1] = 0;
            new_pixels[(j*x_dim + i)*3+2] = 0;
         } else {
            new_pixels[(j*x_dim + i)*3] = pixels[(j*x_dim + i)];
            new_pixels[(j*x_dim + i)*3+1] = pixels[(j*x_dim + i)];
            new_pixels[(j*x_dim + i)*3+2] = pixels[(j*x_dim + i)];
         }
      }
   }

   FILE* outfile = fopen(filename, "wb");
   if (!outfile) {
      fprintf(stderr, "Error opening output JPEG file.\n");
      return;
   }

   cinfo.err = jpeg_std_error(&jerr);
   jpeg_create_compress(&cinfo);
   jpeg_stdio_dest(&cinfo, outfile);

   cinfo.image_width = x_dim;
   cinfo.image_height = y_dim;
   cinfo.input_components = 3; // Color image has three components: Red, Green, and Blue
   cinfo.in_color_space = JCS_RGB; // Set color space to RGB

   jpeg_set_defaults(&cinfo);
   jpeg_set_quality(&cinfo, 75, TRUE);

   jpeg_start_compress(&cinfo, TRUE);
   JSAMPROW row_pointer[1];
   int row_stride = x_dim * 3; // Stride is the number of bytes in a row (width * 3 for RGB)

   while (cinfo.next_scanline < cinfo.image_height) {
      row_pointer[0] = &new_pixels[cinfo.next_scanline * row_stride];
      jpeg_write_scanlines(&cinfo, row_pointer, 1);
   }

   jpeg_finish_compress(&cinfo);
   fclose(outfile);
   jpeg_destroy_compress(&cinfo);
   delete[]new_pixels;
}
