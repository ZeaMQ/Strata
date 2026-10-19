// tests/core/tp_host_runtime_test.cu - the reduce the layer code calls, its refusals, and (with two cards) a
// real collective through it.  Kept to milliseconds of GPU time per case.
#include "strata/core/tp_host_runtime.hpp"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;

static int failed = 0;
#define CHECK(x)                                                             \
    do {                                                                     \
        if (!(x)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ++failed;                                                        \
        }                                                                    \
    } while (0)

int main() {
    std::string err;

    // ---- the refusals, with no pair at all: a rank that cannot reduce must not carry on with half a sum ----
    {
        TpHostRuntime none(nullptr, 0);
        float y[8] = {};
        CHECK(!none.reduce(y, 8, nullptr, err));
        CHECK(err.find("no collective") != std::string::npos);
    }

    // ---- a real pair on this machine's two cards ----------------------------------------------------------
    strata::kernels::ArPair pair;
    int ordinals[strata::kernels::kArRanks] = {0, 1};
    const int64_t width = 640;   // the residual width a layer reduces
    if (!pair.init(ordinals, width, strata::kernels::ArWire::F32, strata::kernels::kArBlocks, err)) {
        std::fprintf(stderr, "SKIP: no two-card collective here: %s\n", err.c_str());
        return 77;   // the project's skip code
    }
    std::printf("tp_host_runtime_test: pair ready on %d blocks, %lld floats\n", pair.blocks(),
                (long long) pair.max_floats());

    TpHostRuntime r0(&pair, 0), r1(&pair, 1);
    {
        // a count past the staging the pair was built for
        float y[8] = {};
        CHECK(!r0.reduce(y, width + 1, nullptr, err));
        CHECK(err.find("staging for") != std::string::npos);
        // and a rank that is not one of the two
        TpHostRuntime bad(&pair, 7);
        CHECK(!bad.reduce(y, 8, nullptr, err));
        CHECK(err.find("rank 7") != std::string::npos);
        // zero floats is not a reduction either
        CHECK(!r0.reduce(y, 0, nullptr, err));
    }

    // ---- and a real collective: two host threads, each its own device, the sum landing in both -----------
    // Each rank starts with its own half (rank 0: 1.0, rank 1: 2.0) and both must end with the total (3.0).
    {
        const int64_t n = 64;
        cudaStream_t s[2] = {nullptr, nullptr};
        float* d[2] = {nullptr, nullptr};
        bool ok = true;
        for (int r = 0; r < strata::kernels::kArRanks; ++r) {
            ok = ok && cudaSetDevice(ordinals[r]) == cudaSuccess;
            ok = ok && cudaStreamCreate(&s[r]) == cudaSuccess;
            ok = ok && cudaMalloc(&d[r], (size_t) n * sizeof(float)) == cudaSuccess;
            std::vector<float> host((size_t) n, r == 0 ? 1.0f : 2.0f);
            ok = ok && cudaMemcpy(d[r], host.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        }
        CHECK(ok);

        std::string e0, e1;
        bool done0 = false, done1 = false;
        std::thread t0([&] {
            CHECK(cudaSetDevice(ordinals[0]) == cudaSuccess);
            done0 = r0.reduce(d[0], n, s[0], e0);
        });
        std::thread t1([&] {
            CHECK(cudaSetDevice(ordinals[1]) == cudaSuccess);
            done1 = r1.reduce(d[1], n, s[1], e1);
        });
        t0.join();
        t1.join();
        if (!done0) std::fprintf(stderr, "rank 0 reduce: %s\n", e0.c_str());
        if (!done1) std::fprintf(stderr, "rank 1 reduce: %s\n", e1.c_str());
        CHECK(done0 && done1);
        for (int r = 0; r < strata::kernels::kArRanks; ++r) {
            CHECK(cudaSetDevice(ordinals[r]) == cudaSuccess);
            CHECK(cudaStreamSynchronize(s[r]) == cudaSuccess);
            std::vector<float> host((size_t) n, 0.0f);
            CHECK(cudaMemcpy(host.data(), d[r], (size_t) n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (int64_t i = 0; i < n; ++i) {
                if (host[(size_t) i] != 3.0f) {
                    ++failed;
                    std::fprintf(stderr, "FAIL rank %d element %lld is %f, not 3.0\n", r, (long long) i,
                                 (double) host[(size_t) i]);
                    break;
                }
            }
            cudaFree(d[r]);
            cudaStreamDestroy(s[r]);
        }
        std::printf("tp_host_runtime_test: both ranks ended at 3.0\n");
    }
    pair.shutdown();

    if (failed == 0) std::printf("tp_host_runtime_test: ok\n");
    return failed ? 1 : 0;
}
