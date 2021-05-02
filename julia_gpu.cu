#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <string.h>
#include <IL/il.h>
#include <IL/ilu.h>
#include <time.h>

using namespace std;

#define N 4096
#define SQRT_2 1.4142
#define MAX_ITER 512

__host__ __device__ void HSVtoRGB( float *r, float *g, float *b, float h, float s, float v );
void saveImage(int width, int height, unsigned char * bitmap, complex<float> seed, int flag);
void compute_julia_CPU(complex<float> c, unsigned char * image);
void compute_julia_GPU(complex<float> c, unsigned char * image);
bool compare_CPU_GPU(unsigned char *image_CPU, unsigned char *image_GPU);
__global__ void julia_kernel(unsigned char * image);

// complex number c is declared as array of float, [real, imaginary] in the constant memory.
__constant__ float d_c[2];

int main(int argc, char **argv)
{
	complex<float> c(0.285f, 0.01f);
	if(argc > 2)
	{
		c.real(atof(argv[1]));
		c.imag(atof(argv[2]));
	} else
		fprintf(stderr, "Usage: %s <real> <imag>\nWhere <real> and <imag> form the complex seed for the Julia set.\n", argv[0]);

	ilInit();
	unsigned char *image_CPU_host = new unsigned char[N*N*3]; //RGB image
	unsigned char *image_GPU_host = new unsigned char[N*N*3]; //RGB image

	const clock_t begin_time = clock();
	compute_julia_CPU(c, image_CPU_host);
	float runTime = (float)(clock() - begin_time) / CLOCKS_PER_SEC;
	printf("Time for julia set computation CPU: %f seconds \n",runTime );

	compute_julia_GPU(c, image_GPU_host);

	bool result = compare_CPU_GPU(image_CPU_host, image_GPU_host);
	fprintf(stderr, "CPU-GPU results do %smatch!\n", (result)?"":"not ");
	// flag is 0 for the cpu image and 1 for gpu image
	saveImage(N, N, image_CPU_host, c, 0);
	saveImage(N, N, image_GPU_host, c, 1);
	delete[] image_CPU_host;
	delete[] image_GPU_host;
}

void compute_julia_CPU(complex<float> c, unsigned char * image)
{
	complex<float> z_old(0.0f, 0.0f);
	complex<float> z_new(0.0f, 0.0f);
	for(int y=0; y<N; y++)
		for(int x=0; x<N; x++)
		{
			z_new.real(4.0f * x / (N) - 2.0f);
			z_new.imag(4.0f * y / (N) - 2.0f);
			int i;
			for(i=0; i<MAX_ITER; i++)
			{
				z_old.real(z_new.real());
				z_old.imag(z_new.imag());
				z_new = pow(z_new, 2);
				z_new += c;
				if(norm(z_new) > 4.0f) break;
			}
			float brightness = (i<MAX_ITER) ? 1.0f : 0.0f;
			float hue = (i % MAX_ITER)/float(MAX_ITER - 1);
			hue = (120*sqrtf(hue) + 150);
			float r, g, b;
			HSVtoRGB(&r, &g, &b, hue, 1.0f, brightness);
			image[(x + y*N)*3 + 0] = (unsigned char)(b*255);
			image[(x + y*N)*3 + 1] = (unsigned char)(g*255);
			image[(x + y*N)*3 + 2] = (unsigned char)(r*255);
		}
}	

void compute_julia_GPU(complex<float> c, unsigned char * image) {
	cudaEvent_t begin, begin_kernel, stop_kernel, stop;
	cudaEventCreate(&begin);
	cudaEventCreate(&begin_kernel);
	cudaEventCreate(&stop_kernel);
	cudaEventCreate(&stop);
	unsigned char* device_image;

	cudaMalloc( (void**) &device_image, N*N*3*sizeof(unsigned char));
	float h_c[] = {c.real(), c.imag()};

	dim3 grid, block;
	block.x = 32;	
	block.y = 32;
	grid.x = N / block.x;
	grid.y = N / block.y;

	cudaEventRecord(begin);
	cudaMemcpyToSymbol(d_c, &h_c, 2*sizeof(float));

	cudaEventRecord(begin_kernel);
	julia_kernel<<<grid, block>>>(device_image);
	cudaEventRecord(stop_kernel);

	cudaMemcpy(image, device_image, N*N*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop_kernel);
	cudaEventSynchronize(stop);

	float kernelTime, totalTime; // Initialize elapsedTime;
	cudaEventElapsedTime(&kernelTime, begin_kernel, stop_kernel);
	cudaEventElapsedTime(&totalTime, begin, stop);
	printf("Time of KERNEL for julia set calculation is: %fms\n", kernelTime);
	printf("Total time for julia set calculation is: %fms\n", totalTime);

	cudaFree(device_image);
}

__global__ void julia_kernel(unsigned char * device_image){
	float z_new_real = 0.0f;
	float z_new_imag = 0.0f;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >= N || y >= N){
		return;
	}
	z_new_real = (4.0f * x / (N) - 2.0f);
	z_new_imag = (4.0f * y / (N) - 2.0f);
	int i;
	float norm_z;
	float temp;
	for(i = 0; i<MAX_ITER; i++){
		// (a + ib)^2 = (a^2 - b^2) + i(2ab)
		temp = z_new_real;
		z_new_real = pow(z_new_real, 2) - pow(z_new_imag, 2);
		z_new_imag = 2 * temp * z_new_imag;
		z_new_real += d_c[0];
		z_new_imag += d_c[1];
		norm_z = pow(z_new_real, 2) + pow(z_new_imag, 2);
		if(norm_z > 4.0f){
			break;
		}
	}
	float brightness = (i<MAX_ITER) ? 1.0f : 0.0f;
	float hue = (i%MAX_ITER)/float(MAX_ITER - 1);
	hue = (120*sqrtf(hue) + 150);
	float r, g, b;
	HSVtoRGB(&r, &g, &b, hue, 1.0f, brightness);
	device_image[(x + y*N)*3 + 0] = (unsigned char)(b*255);
	device_image[(x + y*N)*3 + 1] = (unsigned char)(g*255);
	device_image[(x + y*N)*3 + 2] = (unsigned char)(r*255);
}

//Returns true if GPU results match CPU results, else returns false
bool compare_CPU_GPU(unsigned char *image_CPU, unsigned char *image_GPU)
{
  bool result = true;
  int nelem = N*N*3;
  int count = 0;
  float average_diff = 0.0f;
  for (int i=0; i<nelem; i++) {
    if (image_CPU[i] != image_GPU[i]){
    	result = false;
    	count++;
    	average_diff += abs(image_CPU[i] - image_GPU[i]);
	}
  }
  float percent_diff = count/float(nelem);
  average_diff = average_diff/float(nelem);
  printf("The CPU and GPU images are %f percent similar.\n", (1.0f-percent_diff)*100);
  printf("percetage of pixels different = %f and average difference is = %f \n", percent_diff * 100, average_diff );
  return result;
}

void saveImage(int width, int height, unsigned char * bitmap, complex<float> seed, int flag)
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(width, height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, bitmap);
	ilEnable(IL_FILE_OVERWRITE);
	char imageName[256];
	if(flag == 1){
		sprintf(imageName, "Julia %.3f + i%.3f_gpu.png", seed.real(), seed.imag());		
	}
	else{
		sprintf(imageName, "Julia %.3f + i%.3f_cpu.png", seed.real(), seed.imag());		
	}
	ilSave(IL_PNG, imageName);
	fprintf(stderr, "Image saved as: %s\n", imageName);
}

// r,g,b values are from 0 to 1
// h = [0,360], s = [0,1], v = [0,1]
//		if s == 0, then h = -1 (undefined)
__host__ __device__ void HSVtoRGB( float *r, float *g, float *b, float h, float s, float v )
{
	int i;
	float f, p, q, t;
	if( s == 0 ) {
		// achromatic (grey)
		*r = *g = *b = v;
		return;
	}
	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			*r = v;
			*g = t;
			*b = p;
			break;
		case 1:
			*r = q;
			*g = v;
			*b = p;
			break;
		case 2:
			*r = p;
			*g = v;
			*b = t;
			break;
		case 3:
			*r = p;
			*g = q;
			*b = v;
			break;
		case 4:
			*r = t;
			*g = p;
			*b = v;
			break;
		default:		// case 5:
			*r = v;
			*g = p;
			*b = q;
			break;
	}
}
