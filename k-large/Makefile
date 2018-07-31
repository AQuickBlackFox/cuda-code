CC=/usr/local/cuda/bin/nvcc
ARCH=sm_70
CFLAGS=-arch=$(ARCH) -std=c++11

all: v2 v4 v5 create_bin

create_bin:
	mkdir -p bin

v2: small_gemm_v2.cu
	$(CC) $(CFLAGS) $< -o ./bin/$@

v4: small_gemm_v4.cu
	$(CC) $(CFLAGS) $< -o ./bin/$@

v5: small_gemm_v5.cu
	$(CC) $(CFLAGS) $< -o ./bin/$@

clean:
	rm -r ./bin

test: all
	./bin/v2; ./bin/v4; ./bin/v5
