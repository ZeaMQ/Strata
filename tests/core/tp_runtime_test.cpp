// tests/core/tp_runtime_test.cpp - the seam that decides whether a partial sum is owed. No GPU, no model.
//
// The default path's byte-identity rests on this: a single-card run must never take the reduce branch, and the
// only way that is true is that `current()` is null unless something installs a runtime. That is a small
// property, but it is the one the whole "opt-in, the default path is untouched" claim stands on - so it is
// checked rather than assumed, together with the interface a call site uses.
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "strata/core/tp_runtime.hpp"

using namespace strata::core;

static int failed = 0;
#define CHECK(x)                                                             \
    do {                                                                     \
        if (!(x)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ++failed;                                                        \
        }                                                                    \
    } while (0)

/// What a call site sees: a runtime that records the reductions asked of it.  Also the shape of the real one -
/// `reduce` must be able to fail, because a rank that cannot meet its peer must not carry on with half a sum.
class FakeRuntime : public TpRuntime {
public:
    bool reduce(float* y, int64_t n, void* stream, std::string& err) override {
        (void) y;
        (void) stream;
        ++calls;
        last_n = n;
        if (fail) {
            err = "the peer did not answer";
            return false;
        }
        return true;
    }
    int calls = 0;
    int64_t last_n = 0;
    bool fail = false;
};

int main() {
    std::string err;

    // ---- the default: nothing is owed, and the branch a call site guards on is not taken ---------------
    CHECK(TpRuntime::current() == nullptr);
    // this is the exact shape of the two call sites in layer.cpp
    if (TpRuntime* rt = TpRuntime::current()) {
        std::vector<float> y(4, 0.0f);
        CHECK(rt->reduce(y.data(), 4, nullptr, err));
        CHECK(false);   // unreachable on a single-card run, which is the point
    }

    // ---- an installed runtime is what the layer code calls, with the width it was given ----------------
    {
        FakeRuntime fake;
        TpRuntime::install(&fake);
        CHECK(TpRuntime::current() == &fake);
        std::vector<float> y(640, 1.0f);
        CHECK(TpRuntime::current()->reduce(y.data(), (int64_t) y.size(), nullptr, err));
        CHECK(fake.calls == 1 && fake.last_n == 640);
        // twice a layer, every layer: the call sites are independent, so two calls is two calls
        CHECK(TpRuntime::current()->reduce(y.data(), (int64_t) y.size(), nullptr, err));
        CHECK(fake.calls == 2);
        // ... and a failure is reported rather than swallowed: half a sum looks plausible
        fake.fail = true;
        CHECK(!TpRuntime::current()->reduce(y.data(), (int64_t) y.size(), nullptr, err));
        CHECK(err == "the peer did not answer");
        // uninstalling restores the default path exactly
        TpRuntime::install(nullptr);
        CHECK(TpRuntime::current() == nullptr);
    }

    // ---- and the default is not a one-way door: installing after clearing works the same -----------------
    {
        FakeRuntime a, b;
        TpRuntime::install(&a);
        CHECK(TpRuntime::current() == &a);
        TpRuntime::install(&b);
        CHECK(TpRuntime::current() == &b);
        TpRuntime::install(nullptr);
        CHECK(TpRuntime::current() == nullptr);
    }

    // ---- THE TWO RANKS ARE TWO THREADS IN ONE PROCESS, so what one installs must not be what the other ----
    // This is the property that makes the engine's design possible at all: if the pointer were process-wide,
    // rank 1's install would shadow rank 0's and rank 0 would reduce into rank 1's collective.
    {
        FakeRuntime rank0, rank1;
        bool seen_by_rank0 = false, seen_by_rank1 = false, other_thread_installed = false;
        TpRuntime::install(&rank0);                 // the main thread is rank 0
        std::thread peer([&] {
            TpRuntime::install(&rank1);             // the second rank's own thread
            seen_by_rank1 = TpRuntime::current() == &rank1;
            other_thread_installed = TpRuntime::current() == &rank0;
        });
        peer.join();
        seen_by_rank0 = TpRuntime::current() == &rank0;   // and rank 0 still sees its own
        CHECK(seen_by_rank1);
        CHECK(!other_thread_installed);
        CHECK(seen_by_rank0);
        TpRuntime::install(nullptr);
    }

    if (failed == 0) std::printf("tp_runtime_test: ok\n");
    return failed ? 1 : 0;
}
