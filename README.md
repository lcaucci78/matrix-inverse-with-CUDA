# matrix-inverse-with-CUDA
A CUDA implementation of an in-place matrix inversion algorithm based on Gauss-Jordan elimination is provided.

I do not claim this is the most efficient and numerically accurate implementation out there, but it does the job for me.  Let me know how to make my code better :grinning:

To compile with NVCC, just run:

```
nvcc gpu_matr_inv.cu
```

The implementation is based on the paper by Debabrata DasGupta, "[In-Place Matrix Inversion by Modified Gauss-Jordan Algorithm](https://www.scirp.org/pdf/AM_2013100413422038.pdf)," *Applied Mathematics*, vol. 4, p. 1392-1396, 2013.
