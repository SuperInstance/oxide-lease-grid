//! # oxide-lease-grid
//!
//! Grid-based lease matrix for multi-GPU resource allocation.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState { Leased = 1, Reserved = 0, Free = -1 }

#[derive(Debug, Clone)]
pub struct LeaseGrid {
    width: usize,
    height: usize,
    cells: Vec<CellState>,
    owners: Vec<Option<String>>,
}

impl LeaseGrid {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self { width, height, cells: vec![CellState::Free; size], owners: vec![None; size] }
    }

    fn idx(&self, x: usize, y: usize) -> usize { y * self.width + x }

    pub fn lease(&mut self, x: usize, y: usize, owner: &str) -> bool {
        if x >= self.width || y >= self.height { return false; }
        let i = self.idx(x, y);
        if self.cells[i] != CellState::Free { return false; }
        self.cells[i] = CellState::Leased;
        self.owners[i] = Some(owner.into());
        true
    }

    pub fn reserve(&mut self, x: usize, y: usize, owner: &str) -> bool {
        if x >= self.width || y >= self.height { return false; }
        let i = self.idx(x, y);
        if self.cells[i] != CellState::Free { return false; }
        self.cells[i] = CellState::Reserved;
        self.owners[i] = Some(owner.into());
        true
    }

    pub fn release(&mut self, x: usize, y: usize) -> bool {
        if x >= self.width || y >= self.height { return false; }
        let i = self.idx(x, y);
        self.cells[i] = CellState::Free;
        self.owners[i] = None;
        true
    }

    pub fn get(&self, x: usize, y: usize) -> CellState {
        if x >= self.width || y >= self.height { return CellState::Free; }
        self.cells[self.idx(x, y)]
    }

    pub fn owner(&self, x: usize, y: usize) -> Option<&str> {
        if x >= self.width || y >= self.height { return None; }
        self.owners[self.idx(x, y)].as_deref()
    }

    /// Allocate a rectangular block. Returns top-left corner or None.
    pub fn allocate_block(&mut self, w: usize, h: usize, owner: &str) -> Option<(usize, usize)> {
        for y in 0..=self.height.saturating_sub(h) {
            for x in 0..=self.width.saturating_sub(w) {
                let mut fits = true;
                for dy in 0..h {
                    for dx in 0..w {
                        if self.get(x + dx, y + dy) != CellState::Free { fits = false; break; }
                    }
                    if !fits { break; }
                }
                if fits {
                    for dy in 0..h {
                        for dx in 0..w { self.lease(x + dx, y + dy, owner); }
                    }
                    return Some((x, y));
                }
            }
        }
        None
    }

    /// Fragmentation: 1 - (largest_free_block / total_free)
    pub fn fragmentation(&self) -> f64 {
        let total_free: usize = self.cells.iter().filter(|&&c| c == CellState::Free).count();
        if total_free == 0 { return 0.0; }
        // Simple largest contiguous free row
        let mut max_run = 0usize;
        for y in 0..self.height {
            let mut run = 0usize;
            for x in 0..self.width {
                if self.cells[self.idx(x, y)] == CellState::Free { run += 1; max_run = max_run.max(run); }
                else { run = 0; }
            }
        }
        1.0 - (max_run as f64 / total_free as f64)
    }

    pub fn utilization(&self) -> f64 {
        let leased = self.cells.iter().filter(|&&c| c == CellState::Leased).count();
        leased as f64 / self.cells.len() as f64
    }

    pub fn leased_count(&self) -> usize { self.cells.iter().filter(|&&c| c == CellState::Leased).count() }
    pub fn reserved_count(&self) -> usize { self.cells.iter().filter(|&&c| c == CellState::Reserved).count() }
    pub fn free_count(&self) -> usize { self.cells.iter().filter(|&&c| c == CellState::Free).count() }
    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lease() {
        let mut g = LeaseGrid::new(4, 4);
        assert!(g.lease(0, 0, "a"));
        assert_eq!(g.get(0, 0), CellState::Leased);
        assert_eq!(g.owner(0, 0), Some("a"));
    }

    #[test]
    fn test_double_lease_fails() {
        let mut g = LeaseGrid::new(4, 4);
        g.lease(0, 0, "a");
        assert!(!g.lease(0, 0, "b")); // already leased
    }

    #[test]
    fn test_reserve_release() {
        let mut g = LeaseGrid::new(4, 4);
        g.reserve(1, 1, "a");
        assert_eq!(g.get(1, 1), CellState::Reserved);
        g.release(1, 1);
        assert_eq!(g.get(1, 1), CellState::Free);
    }

    #[test]
    fn test_block_alloc() {
        let mut g = LeaseGrid::new(8, 8);
        let pos = g.allocate_block(3, 2, "kernel_a");
        assert!(pos.is_some());
        assert_eq!(g.leased_count(), 6);
    }

    #[test]
    fn test_block_no_fit() {
        let mut g = LeaseGrid::new(2, 2);
        assert!(g.allocate_block(3, 1, "a").is_none());
    }

    #[test]
    fn test_utilization() {
        let mut g = LeaseGrid::new(4, 4);
        g.lease(0, 0, "a");
        g.lease(1, 0, "a");
        assert!((g.utilization() - 0.125).abs() < 0.01);
    }

    #[test]
    fn test_fragmentation() {
        let mut g = LeaseGrid::new(4, 4);
        // All free = no fragmentation
        assert_eq!(g.fragmentation(), 0.0);
    }

    #[test]
    fn test_counts() {
        let mut g = LeaseGrid::new(4, 4);
        g.lease(0, 0, "a");
        g.reserve(1, 0, "b");
        assert_eq!(g.leased_count(), 1);
        assert_eq!(g.reserved_count(), 1);
        assert_eq!(g.free_count(), 14);
    }
}
