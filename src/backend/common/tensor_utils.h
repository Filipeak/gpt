#pragma once

#define TENSOR_IDX_2D(i, j, dim_j) ((i) * (dim_j) + (j))
#define TENSOR_IDX_3D(i, j, k, dim_j, dim_k) ((i) * (dim_j) * (dim_k) + (j) * (dim_k) + (k))
#define TENSOR_IDX_4D(i, j, k, l, dim_j, dim_k, dim_l) ((i) * (dim_j) * (dim_k) * (dim_l) + (j) * (dim_k) * (dim_l) + (k) * (dim_l) + (l))