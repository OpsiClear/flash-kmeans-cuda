// flash_kmeans_cuda/csrc/assign/assign_policy.cu
//
// Variant catalog + static policy table. Stage 1 transcribes today's
// dispatch order into table form WITHOUT adding new kernel instantiations.

#include "assign_policy.h"
// Kernel template definition needed to instantiate VariantSpec::try_launch<T>
// function pointers. Must come before the constexpr Variant definitions below.
#include "assign_sm80_kernel.cuh"

#include <array>
#include <cstdlib>
#include <cstring>

namespace fkc {
namespace assign {

// =========================================================================
// Variant catalog — every kernel shape currently instantiated.
// =========================================================================

// Wide tiles, BLOCK_N=128, BLOCK_K=128.
constexpr Variant V_WIDEK128_W8         = make_variant<128, 128, 8, 2, 1>("widek128_w8");
constexpr Variant V_WIDEK128_W8_D128    = make_variant<128, 128, 8, 2, 1, 128, true>("widek128_w8_d128");
constexpr Variant V_WIDEK128_W8_N2      = make_variant<128, 128, 8, 2, 2>("widek128_w8_n2");
constexpr Variant V_WIDEK128_W8_N2_D64  = make_variant<128, 128, 8, 2, 2,  64, true>("widek128_w8_n2_d64");
constexpr Variant V_WIDEK128_W8_N2_D96  = make_variant<128, 128, 8, 2, 2,  96, true>("widek128_w8_n2_d96");
constexpr Variant V_WIDEK128_W8_N2_D128 = make_variant<128, 128, 8, 2, 2, 128, true>("widek128_w8_n2_d128");
constexpr Variant V_WIDEK128_W8_N4      = make_variant<128, 128, 8, 2, 4>("widek128_w8_n4");
constexpr Variant V_WIDEK128_W8_N4_D128 = make_variant<128, 128, 8, 2, 4, 128, true>("widek128_w8_n4_d128");

// Wide tiles, BLOCK_N=128, BLOCK_K=96.
constexpr Variant V_WIDEK96_W8          = make_variant<128,  96, 8, 2, 1>("widek96_w8");
constexpr Variant V_WIDEK96_W8_D128     = make_variant<128,  96, 8, 2, 1, 128, true, 256>("widek96_w8_d128");
constexpr Variant V_WIDEK96_W8_N2       = make_variant<128,  96, 8, 2, 2>("widek96_w8_n2");
constexpr Variant V_WIDEK96_W8_N2_D128  = make_variant<128,  96, 8, 2, 2, 128, true, 256>("widek96_w8_n2_d128");
constexpr Variant V_WIDEK96_W8_N4       = make_variant<128,  96, 8, 2, 4>("widek96_w8_n4");
constexpr Variant V_WIDEK96_W8_N4_D128  = make_variant<128,  96, 8, 2, 4, 128, true, 256>("widek96_w8_n4_d128");

// 3-stage wide tiles.
constexpr Variant V_WIDE_3_W8           = make_variant<128,  64, 8, 3, 1>("wide_3_w8");
constexpr Variant V_WIDE_3_W8_N2        = make_variant<128,  64, 8, 3, 2>("wide_3_w8_n2");
constexpr Variant V_WIDE_3_W4           = make_variant<128,  64, 4, 3, 1>("wide_3_w4");
constexpr Variant V_WIDE_2_W4           = make_variant<128,  64, 4, 2, 1>("wide_2_w4");

// Narrow tiles.
constexpr Variant V_NARROW_4            = make_variant< 64,  64, 4, 4, 1>("narrow_4");
constexpr Variant V_NARROWK32_W4        = make_variant< 64,  32, 4, 2, 1>("narrowk32_w4");
constexpr Variant V_NARROWK32_W4_N2     = make_variant< 64,  32, 4, 2, 2>("narrowk32_w4_n2");
constexpr Variant V_NARROWK32_W4_N2_D192= make_variant< 64,  32, 4, 2, 2, 192, true>("narrowk32_w4_n2_d192");
constexpr Variant V_NARROWK32_W4_N2_D224= make_variant< 64,  32, 4, 2, 2, 224, true>("narrowk32_w4_n2_d224");
constexpr Variant V_NARROWK32_W4_N2_D256= make_variant< 64,  32, 4, 2, 2, 256, true>("narrowk32_w4_n2_d256");
constexpr Variant V_NARROWK32_W4_N2_D320= make_variant< 64,  32, 4, 2, 2, 320, true>("narrowk32_w4_n2_d320");
constexpr Variant V_NARROWK32_W4_N2_D384= make_variant< 64,  32, 4, 2, 2, 384, true>("narrowk32_w4_n2_d384");
constexpr Variant V_NARROWK32_W4_N4     = make_variant< 64,  32, 4, 2, 4>("narrowk32_w4_n4");

// Deep tile.
constexpr Variant V_DEEP_2_W4           = make_variant< 64, 128, 4, 2, 1>("deep_2_w4");

// === Stage 2 additions ===================================================
// Wide BK=96 D-specialized (n_tiles=1 and n_tiles=2 each).
constexpr Variant V_WIDEK96_W8_D64       = make_variant<128,  96, 8, 2, 1,  64, true>("widek96_w8_d64");
constexpr Variant V_WIDEK96_W8_D96       = make_variant<128,  96, 8, 2, 1,  96, true>("widek96_w8_d96");
constexpr Variant V_WIDEK96_W8_D192      = make_variant<128,  96, 8, 2, 1, 192, true>("widek96_w8_d192");
constexpr Variant V_WIDEK96_W8_D224      = make_variant<128,  96, 8, 2, 1, 224, true>("widek96_w8_d224");

constexpr Variant V_WIDEK96_W8_N2_D64    = make_variant<128,  96, 8, 2, 2,  64, true>("widek96_w8_n2_d64");
constexpr Variant V_WIDEK96_W8_N2_D96    = make_variant<128,  96, 8, 2, 2,  96, true>("widek96_w8_n2_d96");
constexpr Variant V_WIDEK96_W8_N2_D192   = make_variant<128,  96, 8, 2, 2, 192, true>("widek96_w8_n2_d192");
constexpr Variant V_WIDEK96_W8_N2_D224   = make_variant<128,  96, 8, 2, 2, 224, true>("widek96_w8_n2_d224");

// Wide BK=128 D-specialized (only D=64,96 fit SMEM; D=128 already exists).
constexpr Variant V_WIDEK128_W8_D64      = make_variant<128, 128, 8, 2, 1,  64, true>("widek128_w8_d64");
constexpr Variant V_WIDEK128_W8_D96      = make_variant<128, 128, 8, 2, 1,  96, true>("widek128_w8_d96");

// 3-stage wide BK=64 D-specialized for the larger D values.
constexpr Variant V_WIDE_3_W8_D192       = make_variant<128,  64, 8, 3, 1, 192, true>("wide_3_w8_d192");
constexpr Variant V_WIDE_3_W8_D224       = make_variant<128,  64, 8, 3, 1, 224, true>("wide_3_w8_d224");
constexpr Variant V_WIDE_3_W8_D256       = make_variant<128,  64, 8, 3, 1, 256, true>("wide_3_w8_d256");
constexpr Variant V_WIDE_3_W8_N2_D192    = make_variant<128,  64, 8, 3, 2, 192, true>("wide_3_w8_n2_d192");
constexpr Variant V_WIDE_3_W8_N2_D224    = make_variant<128,  64, 8, 3, 2, 224, true>("wide_3_w8_n2_d224");
constexpr Variant V_WIDE_3_W8_N2_D256    = make_variant<128,  64, 8, 3, 2, 256, true>("wide_3_w8_n2_d256");

// =========================================================================
// Generic-D fallback chain. Used by every cell as the tail of its candidate
// list, and as the entire row for OTHER_D_IDX.
// =========================================================================
constexpr PolicyRow kGenericFallback = {
  &V_WIDEK128_W8, &V_WIDEK96_W8, &V_WIDE_3_W8,
  &V_WIDE_3_W4, &V_NARROW_4, &V_DEEP_2_W4,
  nullptr, nullptr,
};

// =========================================================================
// Per-D ordered candidate lists (Stage 1: transcribes assign_sm80.cu's
// existing if/else for the n_tiles_choice == 2 default path).
// =========================================================================

constexpr PolicyRow kD64 = {
  &V_WIDEK128_W8_N2_D64, &V_WIDEK128_W8_D64, &V_WIDEK96_W8_N2_D64,
  &V_WIDEK96_W8_D64, &V_WIDEK128_W8_N2, &V_WIDEK96_W8_N2,
  &V_WIDE_3_W8, &V_NARROW_4,
};

constexpr PolicyRow kD96 = {
  &V_WIDEK128_W8_N2_D96, &V_WIDEK128_W8_D96, &V_WIDEK96_W8_N2_D96,
  &V_WIDEK96_W8_D96, &V_WIDEK128_W8_N2, &V_WIDEK96_W8_N2,
  &V_WIDE_3_W8, &V_NARROW_4,
};

constexpr PolicyRow kD128 = {
  &V_WIDEK128_W8_N2_D128, &V_WIDEK128_W8_N2, &V_WIDEK96_W8_N2_D128,
  &V_WIDEK96_W8_N2, &V_WIDEK128_W8_D128, &V_WIDEK96_W8_D128,
  &V_WIDE_3_W8, &V_NARROW_4,
};

constexpr PolicyRow kD192 = {
  &V_WIDEK96_W8_N2_D192, &V_WIDEK96_W8_D192, &V_WIDE_3_W8_N2_D192,
  &V_WIDE_3_W8_D192, &V_NARROWK32_W4_N2_D192, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_NARROW_4,
};

constexpr PolicyRow kD224 = {
  &V_WIDEK96_W8_N2_D224, &V_WIDEK96_W8_D224, &V_WIDE_3_W8_N2_D224,
  &V_WIDE_3_W8_D224, &V_NARROWK32_W4_N2_D224, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_NARROW_4,
};

constexpr PolicyRow kD256 = {
  &V_WIDE_3_W8_N2_D256, &V_WIDE_3_W8_D256, &V_NARROWK32_W4_N2_D256,
  &V_NARROWK32_W4_N2, &V_NARROWK32_W4, &V_WIDE_3_W4,
  &V_NARROW_4, &V_DEEP_2_W4,
};

constexpr PolicyRow kD320 = {
  &V_NARROWK32_W4_N2_D320, &V_NARROWK32_W4_N2, &V_NARROWK32_W4,
  &V_WIDE_3_W4, &V_NARROW_4, &V_DEEP_2_W4,
  nullptr, nullptr,
};

constexpr PolicyRow kD384 = {
  &V_NARROWK32_W4_N2_D384, &V_NARROWK32_W4_N2, &V_NARROWK32_W4,
  &V_WIDE_3_W4, &V_NARROW_4, &V_DEEP_2_W4,
  nullptr, nullptr,
};

// Per-D rows are duplicated across all K-buckets in Stage 1 (single static
// order, the same as today's if/else, which only switches on K via the
// prefer_w8 boolean — captured implicitly by ordering w8 variants first).
// kRowsN2 is the n_tiles_choice==2 default path (matches the legacy default).
constexpr PolicyRow kRowsN2[N_D_IDX] = {
  kD64, kD96, kD128, kD192, kD224, kD256, kD320, kD384, kGenericFallback,
};

// =========================================================================
// N_TILES=1 rows — transcribed from legacy n_tiles_choice==1 chain.
// =========================================================================

constexpr PolicyRow kD128_N1 = {
  &V_WIDEK128_W8_D128, &V_WIDEK128_W8, &V_WIDEK96_W8_D128,
  &V_WIDEK96_W8, &V_WIDE_3_W8, &V_WIDE_3_W4,
  &V_NARROW_4, &V_DEEP_2_W4,
};
constexpr PolicyRow kGeneric_N1 = {
  &V_WIDEK128_W8, &V_WIDEK96_W8, &V_WIDE_3_W8,
  &V_WIDE_3_W4, &V_WIDE_2_W4, &V_NARROWK32_W4,
  &V_NARROW_4, &V_DEEP_2_W4,
};

constexpr PolicyRow kRowsN1[N_D_IDX] = {
  kGeneric_N1, kGeneric_N1, kD128_N1, kGeneric_N1, kGeneric_N1,
  kGeneric_N1, kGeneric_N1, kGeneric_N1, kGeneric_N1,
};

// =========================================================================
// N_TILES=4 rows — transcribed from legacy n_tiles_choice==4 branch.
// =========================================================================

constexpr PolicyRow kD128_N4 = {
  &V_WIDEK128_W8_N4_D128, &V_WIDEK128_W8_N4, &V_WIDEK96_W8_N4_D128,
  &V_WIDEK96_W8_N4, &V_NARROWK32_W4_N4, &V_NARROWK32_W4,
  &V_NARROW_4, &V_DEEP_2_W4,
};
constexpr PolicyRow kGeneric_N4 = {
  &V_WIDEK128_W8_N4, &V_WIDEK96_W8_N4, &V_NARROWK32_W4_N4,
  &V_NARROWK32_W4, &V_NARROW_4, &V_DEEP_2_W4,
  nullptr, nullptr,
};

constexpr PolicyRow kRowsN4[N_D_IDX] = {
  kGeneric_N4, kGeneric_N4, kD128_N4, kGeneric_N4, kGeneric_N4,
  kGeneric_N4, kGeneric_N4, kGeneric_N4, kGeneric_N4,
};

VariantView static_policy(int /*dtype_idx*/, int d_idx, int /*k_bucket*/, int n_tiles_override) {
  // Stage 1: identical row regardless of dtype or k_bucket. Stage 3's
  // autotuner reorders within each (dtype,K) cell at runtime.
  // n_tiles_override selects the per-N_TILES row table; 0 uses the N2 default.
  const PolicyRow* row;
  switch (n_tiles_override) {
    case 1:  row = &kRowsN1[d_idx]; break;
    case 4:  row = &kRowsN4[d_idx]; break;
    default: row = &kRowsN2[d_idx]; break;
  }
  return VariantView(row->data(), MAX_CAND);
}

// =========================================================================
// Forced-candidate builder for FKC_* env overrides. Mirrors the legacy
// force_*_env branches.
// =========================================================================
namespace {
constexpr PolicyRow kForcedDeep    = { &V_DEEP_2_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedNarrowN4= { &V_NARROWK32_W4_N4, &V_NARROWK32_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedNarrowN2= { &V_NARROWK32_W4_N2, &V_NARROWK32_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedNarrow  = { &V_NARROWK32_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedW4      = { &V_WIDE_3_W4, &V_WIDE_2_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedWide3N2 = { &V_WIDE_3_W8_N2, &V_WIDE_3_W8, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedWide3   = { &V_WIDE_3_W8, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
}  // namespace

VariantView build_forced_candidates(const EnvKnobs& knobs, const LaunchCtx& ctx) {
  // Precedence mirrors today's force_*_env branches: deep > narrow > w4 > wide3.
  if (knobs.deep) {
    return VariantView(kForcedDeep.data(), MAX_CAND);
  }
  if (knobs.narrow) {
    if (knobs.n_tiles_override == 4) return VariantView(kForcedNarrowN4.data(), MAX_CAND);
    if (knobs.n_tiles_override >= 2) return VariantView(kForcedNarrowN2.data(), MAX_CAND);
    return VariantView(kForcedNarrow.data(), MAX_CAND);
  }
  // NOTE: legacy code, after these forced-shape branches exhausted their
  // candidates on SMEM-too-tight, fell through to the auto-pick chain. The
  // new dispatcher does not — if the forced shape doesn't fit, control
  // falls all the way through to launch_assign_safe. This is the right
  // semantics for an explicit user override; flagged here in case Task 5's
  // equivalence test surfaces a SMEM-tight scenario.
  if (knobs.w4) {
    return VariantView(kForcedW4.data(), MAX_CAND);
  }
  if (knobs.wide3) {
    if (knobs.n_tiles_override >= 2) return VariantView(kForcedWide3N2.data(), MAX_CAND);
    return VariantView(kForcedWide3.data(), MAX_CAND);
  }
  // n_tiles_override on its own: return the per-N_TILES static policy row for
  // the current D and let the dispatch loop pick the first that fits. The
  // legacy code's three separate dispatch tables per n_tiles_choice are
  // preserved via kRowsN1/kRowsN2/kRowsN4.
  return static_policy(/*dtype_idx*/0, d_index_of(ctx.D), k_bucket_of(ctx.K),
                       knobs.n_tiles_override);
}

// =========================================================================
// Env knob reader — called per launch (no static cache).
// =========================================================================
EnvKnobs read_env_knobs() {
  EnvKnobs k;
  if (const char* s = std::getenv("FKC_NTILES")) {
    int v = std::atoi(s);
    if (v == 1 || v == 2 || v == 4) k.n_tiles_override = v;
  }
  auto truthy = [](const char* s) {
    return s && (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0);
  };
  k.wide3    = truthy(std::getenv("FKC_WIDE3"));
  k.w4       = truthy(std::getenv("FKC_W4"));
  k.narrow   = truthy(std::getenv("FKC_NARROW"));
  k.deep     = truthy(std::getenv("FKC_ASSIGN_DEEP_TILE"));
  // FKC_AUTOTUNE defaults to ON. Set FKC_AUTOTUNE=0 to disable.
  if (const char* s = std::getenv("FKC_AUTOTUNE")) {
    k.autotune = !(std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0);
  }
  k.verbose  = truthy(std::getenv("FKC_AUTOTUNE_VERBOSE"));
  return k;
}

}  // namespace assign
}  // namespace fkc
