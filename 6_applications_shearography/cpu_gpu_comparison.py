import time
import matplotlib.pyplot as plt
from math import sqrt

import numpy as np
import cupy as cp

from skimage import data

import skimage.color
from skimage.feature.blob import blob_log
from skimage.feature.blob import blob_dog
from skimage.feature.blob import blob_doh

import cucim.skimage.color
from cucim.skimage.feature import blob_log as blob_log_cu
from cucim.skimage.feature import blob_dog as blob_dog_cu
from cucim.skimage.feature import blob_doh as blob_doh_cu

if __name__ == "__main__":
    gpu_sync = cp.cuda.stream.get_current_stream().synchronize
    _start_total = time.time()
    image = data.hubble_deep_field()

    # CPU COMPUTATION
    _start_cpu = time.time()
    image_cpu = np.asarray(image)
    image_cpu_gray = skimage.color.rgb2gray(image_cpu)

    _a = time.time()
    blobs_log = blob_log(image_cpu_gray, max_sigma=30, num_sigma=10, threshold=.1)
    _runtime_log_cpu = time.time() - _a
    print("elapsed time blob_log: %3.3f seconds" % _runtime_log_cpu)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    _a = time.time()
    blobs_dog = blob_dog(image_cpu_gray, max_sigma=30, threshold=.1)
    _runtime_dog_cpu = time.time() - _a
    print("elapsed time blob_dog: %3.3f seconds" % _runtime_dog_cpu)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    _a = time.time()
    blobs_doh = blob_doh(image_cpu_gray, max_sigma=30, threshold=.1)
    _runtime_doh_cpu = time.time() - _a
    print("elapsed time blob_doh: %3.3f seconds" % _runtime_doh_cpu)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)

    _stop_cpu = time.time()
    _total_runtime_cpu = _stop_cpu - _start_cpu
    print("total runtime cpu: %3.3f seconds" % _total_runtime_cpu)

    # GPU COMPUTATION
    gpu_sync()
    _start_gpu = time.time()
    # cupy necessary for cucim
    image = cp.asarray(image)

    # turn to greyscale
    image_gpu_gray = cucim.skimage.color.rgb2gray(image)

    blob_log_cu(image_gpu_gray, max_sigma=30, num_sigma=10, threshold=.1)  # caching in GPU
    _a = time.time()
    blobs_log_cu = blob_log_cu(image_gpu_gray, max_sigma=30, num_sigma=10, threshold=.1)
    gpu_sync()
    _runtime_log_gpu = time.time() - _a
    print("elapsed time blob_log_cu: %3.3f seconds" % _runtime_log_gpu)

    # Compute radii in the 3rd column.
    blobs_log_cu[:, 2] = blobs_log_cu[:, 2] * sqrt(2)

    blob_dog_cu(image_gpu_gray, max_sigma=30, threshold=.1)  # caching in GPU
    _a = time.time()
    blobs_dog_cu = blob_dog_cu(image_gpu_gray, max_sigma=30, threshold=.1)
    gpu_sync()
    _runtime_dog_gpu = time.time() - _a
    print("elapsed time blob_dog_cu: %3.3f seconds" % _runtime_dog_gpu)
    blobs_dog_cu[:, 2] = blobs_dog_cu[:, 2] * sqrt(2)

    blob_doh_cu(image_gpu_gray, max_sigma=30, threshold=.1)  # caching in GPU
    _a = time.time()
    blobs_doh_cu = blob_doh_cu(image_gpu_gray, max_sigma=30, threshold=.1)
    gpu_sync()
    _runtime_doh_gpu = time.time() - _a
    print("elapsed time blob_doh_cu: %3.3f seconds" % _runtime_doh_gpu)
    blobs_doh_cu[:, 2] = blobs_doh_cu[:, 2] * sqrt(2)

    gpu_sync()
    _stop_gpu = time.time()
    _total_runtime_gpu = _stop_gpu - _start_gpu
    print("total runtime gpu: %3.3f seconds" % _total_runtime_gpu)

    gpu_sync()
    _stop_total = time.time()
    print("runtime=", _stop_total - _start_total)

    # time ratio
    print("The LoG Method is %3.3f times faster on the GPU" % (_runtime_log_cpu/_runtime_log_gpu))
    print("The DoG Method is %3.3f times faster on the GPU" % (_runtime_dog_cpu / _runtime_dog_gpu))
    print("The DoH Method is %3.3f times faster on the GPU" % (_runtime_doh_cpu / _runtime_doh_gpu))

    # plot comparison
    x = np.arange(3)
    y1 = [_runtime_log_cpu, _runtime_dog_cpu, _runtime_doh_cpu]
    y2 = [_runtime_log_gpu, _runtime_dog_gpu, _runtime_doh_gpu]
    width = 0.4

    plt.bar(x-0.2, y1, width, color='orange')
    plt.bar(x+0.2, y2, width, color='green')
    plt.xticks(x, ['LoG', 'DoG', 'DoH'])
    plt.xlabel("Blob-Detection Methods")
    plt.ylabel("Runtime [s]")
    plt.legend(["CPU", "GPU"])
    plt.show()
