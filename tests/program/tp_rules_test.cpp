// tests/program/tp_rules_test.cpp - when a tensor-parallel run may start.  Header-only rule, no GPU, no model.
#include <cstdio>
#include <string>

#include "strata/program/tp_rules.hpp"

using namespace strata::program::tp_rules;

static int failed = 0;
#define CHECK(x)                                                             \
    do {                                                                     \
        if (!(x)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ++failed;                                                        \
        }                                                                    \
    } while (0)

/// Everything a legal TP=2 run needs, so each case below changes exactly one thing.
static Options legal() {
    Options o;
    o.tp = 2;
    o.gpus = 2;
    o.serve = true;
    return o;
}

static bool says(const Options& o, const char* needle) { return refusal(o).find(needle) != std::string::npos; }

int main() {
    // ---- the accepting case, and the default path ------------------------------------------------
    CHECK(refusal(legal()).empty());
    // not asked for: nothing to refuse, whatever else is set - the default path must be untouched
    {
        Options off = legal();
        off.tp = 0;
        off.layer_split = true;
        off.batch = true;
        CHECK(refusal(off).empty());
    }

    // ---- every refusal, one flag at a time --------------------------------------------------------
    {
        Options o = legal();
        o.tp = 1;
        CHECK(says(o, "--tp is 2"));
        o.tp = 4;
        CHECK(says(o, "--tp is 2"));

        o = legal();
        o.serve = false;
        CHECK(says(o, "--serve"));

        o = legal();
        o.gpus = 1;
        CHECK(says(o, "needs two cards"));

        // --tp-device: 0 (the A/B) and the default are fine, and a card the run does not have is refused
        o = legal();
        o.second_device = 0;
        CHECK(refusal(o).empty());
        o.second_device = 1;
        CHECK(refusal(o).empty());
        o = legal();
        o.second_device = -1;                       // not given: the next visible card
        CHECK(refusal(o).empty());
        o = legal();
        o.second_device = 2;                        // a third card on a two-card run
        CHECK(says(o, "--tp-device names card 2"));
        CHECK(says(o, "the A/B of the split"));
        o.second_device = 5;
        CHECK(says(o, "names card 5"));

        o = legal();
        o.layer_split = true;
        CHECK(says(o, "pick one"));

        o = legal();
        o.same_backend = false;
        CHECK(says(o, "one backend"));

        o = legal();
        o.weakest_cc_major = 7;
        o.weakest_cc_minor = 0;
        CHECK(says(o, "7.5"));
        CHECK(says(o, "7.0"));
        o.weakest_cc_major = 6;
        o.weakest_cc_minor = 1;
        CHECK(says(o, "7.5"));

        o = legal();
        o.batch = true;
        CHECK(refusal(o).empty());

        o = legal();
        o.pipeline_windows = true;
        CHECK(refusal(o).empty());

        o = legal();
        o.peer_device = true;
        CHECK(refusal(o).empty());

        o = legal();
        o.remote_experts = true;
        CHECK(refusal(o).empty());

        o = legal();
        o.session_files = true;
        CHECK(refusal(o).empty());

        o = legal();
        o.vision = true;
        CHECK(refusal(o).empty());

        o = legal();
        o.mmap_experts = true;
        CHECK(refusal(o).empty());

        o = legal();
        o.kv_resident = true;
        CHECK(refusal(o).empty());

        o = legal();
        o.prompt_cache_every = true;
        CHECK(refusal(o).empty());
    }

    // ---- the rule order: a run that is illegal several ways reports the one that matters first ---------
    {
        // asked for on one card, with a layer split and batch slots: "two cards" comes before the rest,
        // because without a second card none of the others apply.
        Options o = legal();
        o.gpus = 1;
        o.layer_split = true;
        o.batch = true;
        CHECK(says(o, "needs two cards"));
        // a run with every optional feature on reports the FIRST optional conflict, and it is a real one
        Options all = legal();
        all.batch = all.pipeline_windows = all.peer_device = all.vision = true;
        CHECK(refusal(all).empty());
    }

    // ---- the compute-capability rule on its own ---------------------------------------------------
    CHECK(cc_at_least(7, 5));
    CHECK(cc_at_least(8, 6));      // the two RTX 3080s this was built for
    CHECK(cc_at_least(12, 0));
    CHECK(cc_at_least(9, 0));
    CHECK(!cc_at_least(7, 4));
    CHECK(!cc_at_least(7, 0));
    CHECK(!cc_at_least(6, 1));     // the Quadro P1000 in this machine's third slot
    CHECK(!cc_at_least(5, 2));

    if (failed == 0) std::printf("tp_rules_test: ok\n");
    return failed ? 1 : 0;
}
