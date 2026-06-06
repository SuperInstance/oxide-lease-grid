# oxide-lease-grid

Grid-based lease matrix for multi-GPU resource allocation with ternary cell states. {+1=leased, 0=reserved, -1=free}. 2D allocation, compaction, heatmap.

## Overview

# oxide-lease-grid

Grid-based lease matrix for multi-GPU resource allocation.

## Architecture

This crate sits within the **five-layer Oxide Stack**:

| Layer | Crate | Role |
|-------|-------|------|
| 1 | open-parallel | Async runtime (tokio fork) |
| 2 | pincher | "Vector DB as runtime, LLM as compiler" |
| 3 | flux-core | Bytecode VM + A2A agent protocol |
| 4 | cuda-oxide | Flux→MIR→Pliron→NVVM→PTX compiler |
| 5 | cudaclaw | Persistent GPU kernels, warp consensus, SmartCRDT |

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Code | 177 |
| Public API Surface | 16 items |
| License | Apache-2.0 |

## Installation

```toml
[dependencies]
oxide-lease-grid = "0.1.0"
```

## Usage

```rust
use oxide_lease_grid::*;
// See src/lib.rs tests for complete working examples
```

### Key Types

```
- pub enum CellState { Leased = 1, Reserved = 0, Free = -1 }
- pub struct LeaseGrid {
    pub fn new(width: usize, height: usize) -> Self {
    pub fn lease(&mut self, x: usize, y: usize, owner: &str) -> bool {
    pub fn reserve(&mut self, x: usize, y: usize, owner: &str) -> bool {
    pub fn release(&mut self, x: usize, y: usize) -> bool {
    pub fn get(&self, x: usize, y: usize) -> CellState {
    pub fn owner(&self, x: usize, y: usize) -> Option<&str> {
    pub fn allocate_block(&mut self, w: usize, h: usize, owner: &str) -> Option<(usize, usize)> {
    pub fn fragmentation(&self) -> f64 {
```

## Design Philosophy

This crate uses **ternary algebra** (Z₃) where every value is {-1, 0, +1}:

- **+1** → positive signal (healthy, allocated, converged, ready)
- **0** → neutral (pending, balanced, monitoring, degraded)
- **-1** → negative signal (failed, free, diverged, overloaded)

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary neural networks at 60% less power
2. **GPU warp voting** — hardware ballot instructions return ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity (what goes in must come out)

## Testing

```bash
git clone https://github.com/SuperInstance/oxide-lease-grid.git
cd oxide-lease-grid
cargo test
```

## License

Apache-2.0
