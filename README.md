# oxide-lease-grid

Grid-based lease matrix for multi-GPU spatial resource allocation.

## Why This Exists

GPU resources aren't just scalar quantities you can divide with percentages. On a multi-GPU system, allocation is spatial: which GPU, which memory region, which compute unit. A "lease" is a claim on a specific cell in a resource grid. The grid models the spatial nature of GPU allocation — you can't lease the same SM to two kernels simultaneously, and you need contiguous blocks for some operations.

The ternary cell model captures the lifecycle: **Free** (-1, available), **Reserved** (0, claimed but not active), **Leased** (+1, actively in use). This three-state model handles the common pattern where a kernel reserves resources during compilation but doesn't activate them until dispatch. It also prevents double-allocation without locks — a cell transitions atomically from Free to Reserved or Leased.

## Architecture

```
┌─────────────────────────────────────────────┐
│          LeaseGrid (8×8 example)            │
│                                             │
│    0   1   2   3   4   5   6   7           │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐        │
│0 │ L │ L │ L │ F │ F │ F │ F │ F │        │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤        │
│1 │ L │ L │ L │ F │ F │ R │ R │ F │        │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤        │
│2 │ F │ F │ F │ F │ F │ R │ R │ F │        │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤        │
│3 │ F │ F │ F │ F │ F │ F │ F │ F │        │
│  └───┴───┴───┴───┴───┴───┴───┴───┘        │
│                                             │
│  L = Leased (kernel_a, 3×2 block at 0,0)   │
│  R = Reserved (kernel_b, 2×2 block at 1,5) │
│  F = Free                                   │
│                                             │
│  allocate_block(w, h, owner) → (x, y)       │
│  fragmentation() → f64                      │
│  utilization() → f64                        │
└─────────────────────────────────────────────┘

Block Allocation (first-fit):
  Scan row by row, find first contiguous w×h free block
  Convert all cells to Leased with owner tag

Fragmentation Metric:
  1.0 - (longest free row run / total free cells)
  0.0 = all free or no free cells (unfragmented extremes)
  →1.0 = free cells scattered in small runs (fragmented)
```

**Key types:**

- `CellState` — `Leased(+1)`, `Reserved(0)`, `Free(-1)`
- `LeaseGrid` — width × height matrix of cells with owner tracking

## Usage

```rust
use oxide_lease_grid::LeaseGrid;

let mut grid = LeaseGrid::new(8, 8); // 8×8 resource grid

// Allocate a 3×2 block for a kernel
let pos = grid.allocate_block(3, 2, "conv2d_kernel");
assert_eq!(pos, Some((0, 0))); // top-left corner
assert_eq!(grid.leased_count(), 6); // 3 × 2 = 6 cells

// Reserve a 2×2 block (claimed but not active)
grid.reserve(5, 1, "matmul_kernel");
grid.reserve(5, 2, "matmul_kernel");
grid.reserve(6, 1, "matmul_kernel");
grid.reserve(6, 2, "matmul_kernel");

// Check utilization and fragmentation
println!("Utilization: {:.1}%", grid.utilization() * 100.0);
println!("Fragmentation: {:.2}", grid.fragmentation());

// Release cells when done
grid.release(0, 0); // release single cell
assert_eq!(grid.owner(0, 0), None);

// Check counts
println!("Leased: {}, Reserved: {}, Free: {}",
    grid.leased_count(), grid.reserved_count(), grid.free_count());
```

## API Reference

### `CellState`

```rust
pub enum CellState {
    Leased = 1,    // Actively in use
    Reserved = 0,  // Claimed but not active
    Free = -1,     // Available
}
```

### `LeaseGrid`

- `new(width: usize, height: usize) -> Self` — create empty grid
- `lease(x, y, owner) -> bool` — lease a single cell
- `reserve(x, y, owner) -> bool` — reserve a single cell
- `release(x, y) -> bool` — free a cell regardless of state
- `get(x, y) -> CellState` — query cell state (out of bounds = `Leased`)
- `owner(x, y) -> Option<&str>` — query cell owner
- `allocate_block(w, h, owner) -> Option<(usize, usize)>` — first-fit block allocation, returns top-left corner
- `fragmentation() -> f64` — 0.0 = unfragmented, →1.0 = scattered free cells
- `utilization() -> f64` — ratio of leased cells to total
- `leased_count() -> usize` / `reserved_count() -> usize` / `free_count() -> usize`
- `width() -> usize` / `height() -> usize`

## The Deeper Idea

This is the **spatial allocation layer** in the oxide stack's resource management architecture. While oxide-tenancy handles *how much* resource each tenant gets (scalar allocation), oxide-lease-grid handles *where* those resources are (spatial allocation). The grid model maps naturally to real GPU topologies: rows can represent SMs, columns can represent time slices or memory banks.

The ternary cell lifecycle (Free → Reserved → Leased → Free) mirrors the three-phase pattern in hardware resource management. A kernel is compiled (Reserved), dispatched (Leased), and completed (released back to Free). The fragmentation metric tells you when the grid needs compaction — a high fragmentation score means small free regions are scattered between large leased blocks, and you should consider defragmentation before the next large allocation request.

## Related Crates

- **oxide-tenancy** — scalar resource allocation that determines *how much* grid space each tenant needs
- **oxide-federation** — cross-cluster federation that routes work to nodes with available grid space
- **oxide-capacity** — capacity planning informed by grid utilization
- **oxide-checkpoint** — saves grid state for recovery after failures
