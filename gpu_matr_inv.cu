#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>


void GPU_matrix_inverse_test_1();
void GPU_matrix_inverse_test_2();
void GPU_matrix_inverse_test_3();
template <class T> void GPU_matrix_inverse(T *mat_dev, std::size_t N);
template <class T> static __global__ void GPU_matrix_inverse_step_1_kernel(T *mat_dev, unsigned int N, unsigned int p, T *pivots_dev);
template <class T> static __global__ void GPU_matrix_inverse_step_2_kernel(T *mat_dev, unsigned int N, unsigned int p, T *pivots_dev);


template <class T, class = class std::enable_if<std::is_integral<T>::value>::type>
__host__ __device__ __forceinline__ T map(T dim_x, T dim_y, T n_x, T n_y) {
  return(dim_x * n_y + n_x);
}


int main(int argc, char **argv) {
  std::cout << std::fixed << std::setprecision(6);
  GPU_matrix_inverse_test_1();
  GPU_matrix_inverse_test_2();
  GPU_matrix_inverse_test_3();
  return(0);
}


void GPU_matrix_inverse_test_1() {
  const std::size_t N = 3;
  std::vector<double> mat = std::vector<double>(N * N);
  std::vector<double> inv = std::vector<double>(N * N);
  double *mat_dev;
  
  std::cout << "*** TEST 1 ***" << std::endl << std::endl;
  mat[map(3, 3, 0, 0)] = -1.00;		mat[map(3, 3, 0, 1)] = -1.00;		mat[map(3, 3, 0, 2)] = 3.00;
  mat[map(3, 3, 1, 0)] = 2.00;		mat[map(3, 3, 1, 1)] = 1.00;		mat[map(3, 3, 1, 2)] = 2.00;
  mat[map(3, 3, 2, 0)] = -2.00;		mat[map(3, 3, 2, 1)] = -2.00;		mat[map(3, 3, 2, 2)] = 1.00;
  cudaMalloc(& mat_dev, N * N * sizeof(double));
  cudaMemcpy(mat_dev, mat.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
  GPU_matrix_inverse(mat_dev, N);
  cudaMemcpy(inv.data(), mat_dev, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(mat_dev);
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "A:" << std::endl;
  for(std::size_t m = 0; m < N; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      std::cout << mat[map(N, N, m, n)] << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "A^{-1}:" << std::endl;
  for(std::size_t m = 0; m < N; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      std::cout << inv[map(N, N, m, n)] << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "A * A^{-1}:" << std::endl;
  for(std::size_t m = 0; m < N; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      double sum = 0.00;
      for(std::size_t k = 0; k < N; ++k) {
        sum += mat[map(N, N, m, k)] * inv[map(N, N, k, n)];
      }
      std::cout << sum << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "A^{-1} * A:" << std::endl;
  for(std::size_t m = 0; m < N; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      double sum = 0.00;
      for(std::size_t k = 0; k < N; ++k) {
        sum += inv[map(N, N, m, k)] * mat[map(N, N, k, n)];
      }
      std::cout << sum << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  return;
}


void GPU_matrix_inverse_test_2() {
  const std::size_t N = 1000;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distr(-5.00, +5.00);
  std::vector<double> mat = std::vector<double>(N * N);
  std::vector<double> inv = std::vector<double>(N * N);
  double *mat_dev;

  std::cout << "*** TEST 2 ***" << std::endl << std::endl;
  std::cout << "Computing the inverse of a random matrix and checking the result... " << std::flush;
  for(std::size_t m = 0; m < N; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      mat[map(N, N, m, n)] = distr(gen);
    }
  }
  cudaMalloc(& mat_dev, N * N * sizeof(double));
  cudaMemcpy(mat_dev, mat.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
  GPU_matrix_inverse(mat_dev, N);
  cudaMemcpy(inv.data(), mat_dev, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(mat_dev);
  for(std::size_t m = 0; m < N; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      if(!std::isfinite(inv[map(N, N, m, n)])) {
        std::cerr << "Test failed!" << std::endl << std::endl;
        return;
      }
      double sum = 0.00;
      for(std::size_t k = 0; k < N; ++k) {
        sum += mat[map(N, N, m, k)] * inv[map(N, N, k, n)];
      }
      const double expected = (m == n) ? 1.00 : 0.00;
      if(std::fabs(sum - expected) > 0.0001) {
        std::cerr << "Test failed!" << std::endl << std::endl;
        return;
      }
    }
  }
  for(std::size_t m = 0; m < N; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      double sum = 0.00;
      for(std::size_t k = 0; k < N; ++k) {
        sum += inv[map(N, N, m, k)] * mat[map(N, N, k, n)];
      }
      const double expected = (m == n) ? 1.00 : 0.00;
      if(std::abs(sum - expected) > 0.0001) {
        std::cerr << "Test failed!" << std::endl << std::endl;
        return;
      }
    }
  }
  std::cerr << "Test passed!" << std::endl << std::endl;
  return;
}


void GPU_matrix_inverse_test_3() {
  const std::size_t N = 20000;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distr(-5.00, +5.00);
  std::vector<double> mat = std::vector<double>(N * N);
  std::vector<double> inv = std::vector<double>(N * N);
  std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
  std::chrono::duration<double> duration;
  double *mat_dev;

  std::cout << "*** TEST 3 ***" << std::endl << std::endl;
  std::cout << "Timing the calculation of the inverse of a " << N << "x" << N << " random matrix... " << std::flush; 
  for(std::size_t m = 0; m < N; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      mat[map(N, N, m, n)] = distr(gen);
    }
  }
  cudaMalloc(& mat_dev, N * N * sizeof(double));
  cudaMemcpy(mat_dev, mat.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  GPU_matrix_inverse(mat_dev, N);
  cudaDeviceSynchronize();
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  cudaMemcpy(inv.data(), mat_dev, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(mat_dev);
  std::cout << "elapsed time: " << duration.count() << " seconds" << std::endl;
  return;
}


// Adapted from D. DasGupta, "In-Place Matrix Inversion by Modified
// Gauss-Jordan Algorithm," Applied Mathematics, vol. 4, p. 1392-1396, 2013
// Available at: https://www.scirp.org/pdf/AM_2013100413422038.pdf
template <class T>
void GPU_matrix_inverse(T *mat_dev, std::size_t N) {
  const unsigned int block_size_1D = 8;
  const unsigned int num_blocks_1D = (N + block_size_1D - 1) / block_size_1D;
  T *pivots_dev;

  cudaMalloc(& pivots_dev, N * sizeof(T));
  for(std::size_t p = 0; p < N; ++p) {
    cudaMemcpy(& pivots_dev[p], mat_dev + map(N, N, p, p), sizeof(T), cudaMemcpyDeviceToDevice);
    GPU_matrix_inverse_step_1_kernel<<<num_blocks_1D, block_size_1D>>>(mat_dev, N, p, pivots_dev);
    GPU_matrix_inverse_step_2_kernel<<<dim3(num_blocks_1D, num_blocks_1D), dim3(block_size_1D, block_size_1D)>>>(mat_dev, N, p, pivots_dev);
  }
  cudaFree(pivots_dev);
  return;
}


template <class T>
static __global__ void GPU_matrix_inverse_step_1_kernel(T *mat_dev, unsigned int N, unsigned int p, T *pivots_dev) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < N) {
    if(i != p) {
      mat_dev[map(N, N, p, i)] /= pivots_dev[p];
      pivots_dev[i] = mat_dev[map(N, N, i, p)];
      mat_dev[map(N, N, i, p)] = T(0);
    } else {
      mat_dev[map(N, N, p, p)] = T(1) / pivots_dev[p];
    }
  }
  return;
}


template <class T>
static __global__ void GPU_matrix_inverse_step_2_kernel(T *mat_dev, unsigned int N, unsigned int p, T *pivots_dev) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if((i < N) && (j < N)) {
    if(i != p) {
      mat_dev[map(N, N, i, j)] -= mat_dev[map(N, N, p, j)] * pivots_dev[i];
    }
  }
  return;
}
