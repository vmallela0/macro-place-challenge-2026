"""MLX batch evaluator: GPU-accelerated proxy-cost ranking for B candidate moves.

Mirrors IncrementalEvaluator's HPWL and density terms exactly. The congestion
term is approximated for ranking purposes by holding the per-net routing demand
(V_routing_smooth, H_routing_smooth) frozen at the current state and only
updating the macro-blockage contribution per candidate. This is the dominant
single-move signal — a macro moving cell-to-cell mostly shifts blockage rather
than reroutes.

The CPU IncrementalEvaluator is the source of truth for accept-or-reject. This
class only proposes ranked candidates. Any move accepted on a GPU score must
also improve the CPU-exact cost.

Usage
-----
    bench = Benchmark.load(...)
    plc = _load_plc(...)
    incr = IncrementalEvaluator(plc, bench)
    gpu = MLXBatchEvaluator(incr, bench)
    cands = mx.array(np.random.uniform(...).reshape(B, 2))
    scores = gpu.score_candidates(macro_idx=3, candidate_xy=cands)
    # ... pick best, validate on incr.move_macro(...), accept if exact improves
    gpu.notify_committed_move(macro_idx=3, new_x=..., new_y=...)

Math
----
HPWL: for each net n containing macro m, precompute
  other_x_lo[m,n] = min over pins on n that are NOT from m
  other_x_hi[m,n] = max  ditto
  m_xoff_lo[m,n] = min of m's pin x-offsets on n  (m's pin x = cx + xoff)
  m_xoff_hi[m,n] = max  ditto
  (and y-equivalents)
Then for candidate (cx, cy):
  new_lo_x = min(other_x_lo, cx + m_xoff_lo)
  new_hi_x = max(other_x_hi, cx + m_xoff_hi)
  new_hpwl_n = (new_hi_x - new_lo_x) + (new_hi_y - new_lo_y)
  delta_hpwl_n = weight[n] * (new_hpwl_n - current_hpwl_n)
sum over nets touching m → delta total HPWL → new wirelength_cost via norm.

Density: per macro, footprint cells and per-cell areas depend on (cx, cy, w, h).
Vectorize over a (B, max_cells_in_footprint) tile, mask invalid cells, scatter
into a (B, n_cells) delta grid. New grid_density_b = current_grid - delta_old +
delta_new_b. Top-(density_cnt) sum over n_cells per b. Float64 accumulation to
match the CPU evaluator's precision.

Congestion (approx): same scattering for V_macro_raw and H_macro_raw. Then
  V_total_b = V_routing_smooth + (V_macro_raw + dV_b) / grid_v_routes
  H_total_b = H_routing_smooth + (H_macro_raw + dH_b) / grid_h_routes
  cong_b = top_5%_avg(concat(V_total_b, H_total_b))
The smoothed routing demand stays frozen — this is the approximation. The
IncrementalEvaluator validates exact congestion (with rerouting) on commit.
"""
from __future__ import annotations
import math
import numpy as np

try:
    import mlx.core as mx
    _MLX_OK = True
except Exception:
    mx = None
    _MLX_OK = False


def mlx_available() -> bool:
    return _MLX_OK


class MLXBatchEvaluator:
    """GPU batch ranker over candidate single-macro moves.

    Holds a static snapshot of the placement at construction. Call
    `notify_committed_move` after each `incr_eval.move_macro` accept to
    refresh the affected per-net extremes and the macro/density grids.
    """

    def __init__(self, incr_eval, benchmark):
        if not _MLX_OK:
            raise RuntimeError("mlx is not available")

        # Cache references to the CPU evaluator's static structure. We never
        # mutate incr_eval here.
        self.incr = incr_eval
        self.bench = benchmark
        self.n_total = int(incr_eval.macro_pos.shape[0])
        self.n_hard = int(incr_eval.n_hard)
        self.n_nets = int(incr_eval.n_nets)
        self.n_cells = int(incr_eval.n_cells)
        self.grid_col = int(incr_eval.grid_col)
        self.grid_row = int(incr_eval.grid_row)
        self.grid_width = float(incr_eval.grid_width)
        self.grid_height = float(incr_eval.grid_height)
        self.grid_area = float(incr_eval.grid_area)
        self.grid_v_routes = float(incr_eval.grid_v_routes)
        self.grid_h_routes = float(incr_eval.grid_h_routes)
        self.vrouting_alloc = float(incr_eval.vrouting_alloc)
        self.hrouting_alloc = float(incr_eval.hrouting_alloc)
        self.cw = float(incr_eval.cw)
        self.ch = float(incr_eval.ch)
        self.net_cnt_value = float(incr_eval.net_cnt)
        self.density_cnt = int(incr_eval.density_cnt)
        # Top 5% cells over (V_total ∪ H_total), matching _recompute_congestion_cost.
        self.cong_top_cnt = max(1, math.floor(2 * self.n_cells * 0.05))

        # Per-(macro m, net n) extremes precomputation -------------------------
        # Build CSR-style arrays: for each macro, for each net it touches,
        # other-pin x-min/max, y-min/max, and macro pin x/y-offset min/max.
        self._build_macro_net_extremes()

        # Build per-macro footprint geometry (size, half-extent) -----------------
        self.macro_w = mx.array(np.asarray(incr_eval.macro_w, dtype=np.float32))
        self.macro_h = mx.array(np.asarray(incr_eval.macro_h, dtype=np.float32))
        # Static grid state
        self._grid_density_np = np.asarray(incr_eval.grid_density, dtype=np.float64).copy()
        self._V_macro_raw_np = np.asarray(incr_eval.V_macro_raw, dtype=np.float64).copy()
        self._H_macro_raw_np = np.asarray(incr_eval.H_macro_raw, dtype=np.float64).copy()
        self._V_smooth_np = np.asarray(incr_eval.V_routing_smooth, dtype=np.float64).copy()
        self._H_smooth_np = np.asarray(incr_eval.H_routing_smooth, dtype=np.float64).copy()

        self._grid_density_mx = mx.array(self._grid_density_np.astype(np.float32))
        self._V_macro_mx = mx.array(self._V_macro_raw_np.astype(np.float32))
        self._H_macro_mx = mx.array(self._H_macro_raw_np.astype(np.float32))
        self._V_smooth_mx = mx.array(self._V_smooth_np.astype(np.float32))
        self._H_smooth_mx = mx.array(self._H_smooth_np.astype(np.float32))

        # Current totals
        self.total_hpwl = float(incr_eval.total_hpwl)
        # Wirelength normalization factor: cost = total_hpwl / ((cw+ch)*net_cnt)
        self._wl_norm = (self.cw + self.ch) * self.net_cnt_value

    # ------------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------------

    def _build_macro_net_extremes(self):
        """For every macro and every net it touches, precompute the min/max
        of pin x and y coordinates over the OTHER pins on that net (i.e. not
        from this macro), plus the min/max of this macro's own x and y pin
        offsets on this net.

        Stored as flat arrays in CSR layout indexed by macro:
          mn_starts[m]   : start index into the flat per-(m, net) arrays
          mn_starts[m+1] : end index
          mn_net_id      : net id for each (m, net) entry
          mn_other_xlo / xhi / ylo / yhi  : extremes of OTHER pins
          mn_xoff_lo / xoff_hi / yoff_lo / yoff_hi : extremes of m's own offsets
          mn_weight      : net weight
          mn_old_hpwl    : net's hpwl as currently held by incr_eval
                           (used to compute delta in score_candidates)
        """
        incr = self.incr
        n_total = self.n_total

        starts = [0]
        net_ids: list[int] = []
        other_xlo: list[float] = []
        other_xhi: list[float] = []
        other_ylo: list[float] = []
        other_yhi: list[float] = []
        xoff_lo: list[float] = []
        xoff_hi: list[float] = []
        yoff_lo: list[float] = []
        yoff_hi: list[float] = []
        weights: list[float] = []
        old_hpwl: list[float] = []

        pin_x = np.asarray(incr.pin_x, dtype=np.float64)
        pin_y = np.asarray(incr.pin_y, dtype=np.float64)
        pin_macro = np.asarray(incr.pin_macro, dtype=np.int32)
        pin_xoff = np.asarray(incr.pin_xoff, dtype=np.float64)
        pin_yoff = np.asarray(incr.pin_yoff, dtype=np.float64)
        net_starts = np.asarray(incr.net_starts, dtype=np.int32)
        net_weight = np.asarray(incr.net_weight, dtype=np.float64)
        net_hpwl = np.asarray(incr.net_hpwl, dtype=np.float64)

        for m in range(n_total):
            for nid in incr.macro_nets[m]:
                s = int(net_starts[nid])
                e = int(net_starts[nid + 1])
                pm = pin_macro[s:e]
                # Pins from this macro
                mine = (pm == m)
                them = ~mine
                if them.any():
                    oxs = pin_x[s:e][them]
                    oys = pin_y[s:e][them]
                    other_xlo.append(float(oxs.min()))
                    other_xhi.append(float(oxs.max()))
                    other_ylo.append(float(oys.min()))
                    other_yhi.append(float(oys.max()))
                else:
                    # Whole net is this macro — degenerate; HPWL becomes (xoff range
                    # + yoff range), independent of position. Mark with sentinels
                    # that will keep deltas at zero (use this macro's own pin
                    # extremes shifted to current pos).
                    cx = float(incr.macro_pos[m, 0])
                    cy = float(incr.macro_pos[m, 1])
                    xs = pin_x[s:e]
                    ys = pin_y[s:e]
                    other_xlo.append(float(xs.min()))
                    other_xhi.append(float(xs.max()))
                    other_ylo.append(float(ys.min()))
                    other_yhi.append(float(ys.max()))

                # This macro's own pin offsets on this net
                if mine.any():
                    mxoff = pin_xoff[s:e][mine]
                    myoff = pin_yoff[s:e][mine]
                    xoff_lo.append(float(mxoff.min()))
                    xoff_hi.append(float(mxoff.max()))
                    yoff_lo.append(float(myoff.min()))
                    yoff_hi.append(float(myoff.max()))
                else:
                    # Macro is "in" the net via macro_nets bookkeeping but has no
                    # actual pin entries. Use neutral sentinels (offset 0) so
                    # delta_hpwl is dominated by other-pin extremes only.
                    xoff_lo.append(0.0)
                    xoff_hi.append(0.0)
                    yoff_lo.append(0.0)
                    yoff_hi.append(0.0)

                net_ids.append(int(nid))
                weights.append(float(net_weight[nid]))
                old_hpwl.append(float(net_hpwl[nid]))
            starts.append(len(net_ids))

        self.mn_starts = np.asarray(starts, dtype=np.int32)
        self.mn_net_ids = np.asarray(net_ids, dtype=np.int32)

        self._mn_other_xlo_np = np.asarray(other_xlo, dtype=np.float64)
        self._mn_other_xhi_np = np.asarray(other_xhi, dtype=np.float64)
        self._mn_other_ylo_np = np.asarray(other_ylo, dtype=np.float64)
        self._mn_other_yhi_np = np.asarray(other_yhi, dtype=np.float64)
        self._mn_xoff_lo_np = np.asarray(xoff_lo, dtype=np.float64)
        self._mn_xoff_hi_np = np.asarray(xoff_hi, dtype=np.float64)
        self._mn_yoff_lo_np = np.asarray(yoff_lo, dtype=np.float64)
        self._mn_yoff_hi_np = np.asarray(yoff_hi, dtype=np.float64)
        self._mn_weight_np = np.asarray(weights, dtype=np.float64)
        self._mn_old_hpwl_np = np.asarray(old_hpwl, dtype=np.float64)

    # ------------------------------------------------------------------------
    # Scoring API
    # ------------------------------------------------------------------------

    def score_components(self, macro_idx: int, candidate_xy,
                         skip_density: bool = False,
                         skip_congestion: bool = False):
        """Like `score_candidates` but returns (wl_cost, density_cost,
        congestion_cost) per candidate as a tuple of (B,) MLX arrays.

        Each component matches the conventions of `IncrementalEvaluator`:
        - wl_cost = total_hpwl_after / ((cw + ch) * net_cnt)
        - density_cost is the 0.5 * top-density_cnt average of grid_density
        - congestion_cost is the top-5% mean of (V_total ∪ H_total) (no 0.5)
        """
        wl, dens, cong = self._score_components_impl(
            macro_idx, candidate_xy, skip_density, skip_congestion)
        return wl, dens, cong

    def score_candidates(self, macro_idx: int, candidate_xy,
                         skip_density: bool = False,
                         skip_congestion: bool = False):
        """Score B candidate (x,y) positions for a single macro.

        Parameters
        ----------
        macro_idx : int
            The macro to be (hypothetically) moved.
        candidate_xy : mlx.core.array of shape (B, 2) float32
            Candidate centers.
        skip_density, skip_congestion : bool
            If True, that term is excluded from the returned score (useful
            when callers want to score nets only — much faster).

        Returns
        -------
        scores : mlx.core.array of shape (B,) float32
            proxy_cost = wirelength + 0.5 density + 0.5 congestion
            with the conventions of `IncrementalEvaluator.get_proxy_cost`.
        """
        wl, dens, cong = self._score_components_impl(
            macro_idx, candidate_xy, skip_density, skip_congestion)
        proxy = wl + np.float32(0.5) * dens + np.float32(0.5) * cong
        return proxy

    def _score_components_impl(self, macro_idx, candidate_xy,
                               skip_density, skip_congestion):
        cands = candidate_xy
        cx = cands[:, 0]
        cy = cands[:, 1]
        B = int(cands.shape[0])

        # ---------------- HPWL delta ----------------
        s = int(self.mn_starts[macro_idx])
        e = int(self.mn_starts[macro_idx + 1])
        if e > s:
            other_xlo = mx.array(self._mn_other_xlo_np[s:e].astype(np.float32))
            other_xhi = mx.array(self._mn_other_xhi_np[s:e].astype(np.float32))
            other_ylo = mx.array(self._mn_other_ylo_np[s:e].astype(np.float32))
            other_yhi = mx.array(self._mn_other_yhi_np[s:e].astype(np.float32))
            xoff_lo = mx.array(self._mn_xoff_lo_np[s:e].astype(np.float32))
            xoff_hi = mx.array(self._mn_xoff_hi_np[s:e].astype(np.float32))
            yoff_lo = mx.array(self._mn_yoff_lo_np[s:e].astype(np.float32))
            yoff_hi = mx.array(self._mn_yoff_hi_np[s:e].astype(np.float32))
            weights = mx.array(self._mn_weight_np[s:e].astype(np.float32))
            old_hpwl = mx.array(self._mn_old_hpwl_np[s:e].astype(np.float32))

            # Broadcast: (B, 1) op (1, M) → (B, M)
            cxB = cx[:, None]
            cyB = cy[:, None]
            mine_xlo = cxB + xoff_lo[None, :]
            mine_xhi = cxB + xoff_hi[None, :]
            mine_ylo = cyB + yoff_lo[None, :]
            mine_yhi = cyB + yoff_hi[None, :]

            new_xlo = mx.minimum(other_xlo[None, :], mine_xlo)
            new_xhi = mx.maximum(other_xhi[None, :], mine_xhi)
            new_ylo = mx.minimum(other_ylo[None, :], mine_ylo)
            new_yhi = mx.maximum(other_yhi[None, :], mine_yhi)

            new_hpwl = (new_xhi - new_xlo) + (new_yhi - new_ylo)
            delta = weights[None, :] * (new_hpwl - old_hpwl[None, :])
            delta_total_hpwl = mx.sum(delta, axis=1)  # (B,)
        else:
            delta_total_hpwl = mx.zeros((B,), dtype=mx.float32)

        wl_total = mx.array(np.float32(self.total_hpwl)) + delta_total_hpwl
        wl_cost = wl_total / np.float32(self._wl_norm)

        # ---------------- Density delta ----------------
        if not skip_density:
            density_cost = self._density_cost_batch(macro_idx, cx, cy)
        else:
            density_cost = mx.zeros((B,), dtype=mx.float32)

        # ---------------- Congestion delta (approx) ----------------
        if not skip_congestion and macro_idx < self.n_hard:
            cong_cost = self._congestion_cost_batch(macro_idx, cx, cy)
        else:
            # For soft macros we leave congestion at the current value.
            cong_cost = mx.full((B,), float(self.incr.congestion_cost), dtype=mx.float32)

        return wl_cost, density_cost, cong_cost

    # ------------------------------------------------------------------------
    # Density: macro-footprint scatter, top-K sum
    # ------------------------------------------------------------------------

    def _density_cost_batch(self, macro_idx, cx, cy):
        """Compute the new density_cost for B candidate positions of macro
        `macro_idx`. Returns (B,) MLX float32.

        Vectorized: per-candidate footprint cells form a regular tile of
        (n_rows_max x n_cols_max) cells around the candidate; mask out the
        cells outside the actual bl..ur rectangle.
        """
        w = float(self.incr.macro_w[macro_idx])
        h = float(self.incr.macro_h[macro_idx])
        if w <= 0.0 or h <= 0.0:
            return mx.full((cx.shape[0],), float(self.incr.density_cost), dtype=mx.float32)

        old_x = float(self.incr.macro_pos[macro_idx, 0])
        old_y = float(self.incr.macro_pos[macro_idx, 1])
        old_cells, old_areas = self._cell_areas(old_x, old_y, w, h)

        cx_np = np.asarray(cx, dtype=np.float32)
        cy_np = np.asarray(cy, dtype=np.float32)
        B = cx_np.shape[0]

        # Subtract old footprint from a base grid copy.
        base = self._grid_density_np.copy()
        for ci, ai in zip(old_cells, old_areas):
            base[ci] -= ai / self.grid_area

        # Vectorized per-candidate footprint cells -----------------------------
        flat_idx, area_per_cell, valid = self._cell_areas_batch(cx_np, cy_np, w, h)
        # area_per_cell: (B, R*C); flat_idx: (B, R*C); valid: (B, R*C) bool

        full = np.broadcast_to(base.astype(np.float32), (B, self.n_cells)).copy()
        flat_b = np.repeat(np.arange(B), flat_idx.shape[1]).reshape(flat_idx.shape)
        flat_idx_safe = np.where(valid, flat_idx, 0)
        flat_indices = (flat_b * self.n_cells + flat_idx_safe).ravel()
        flat_values = (area_per_cell / np.float32(self.grid_area)).ravel()
        flat_values = np.where(valid.ravel(), flat_values, np.float32(0.0))
        np.add.at(full.ravel(), flat_indices, flat_values)

        full_mx = mx.array(full)
        cnt = self.density_cnt
        # MLX top-k via partition: take top `cnt` per row, sum.
        sorted_full = mx.sort(full_mx, axis=1)
        top = sorted_full[:, -cnt:]
        density_cost = mx.sum(top, axis=1) / np.float32(cnt) * np.float32(0.5)
        return density_cost

    def _cell_areas_batch(self, cx_np, cy_np, w, h):
        """Vectorized version of `_cell_areas` over B candidates of a single
        macro of fixed size (w, h).

        Returns:
          flat_idx : (B, K) int64 — flattened cell indices (0 where invalid)
          areas    : (B, K) float32 — overlap areas (0 where invalid)
          valid    : (B, K) bool
        K = n_rows_max * n_cols_max where n_rows_max = ceil(h/grid_h)+2.
        """
        B = cx_np.shape[0]
        half_w = w / 2.0
        half_h = h / 2.0
        x_min = (cx_np - half_w).astype(np.float64)
        y_min = (cy_np - half_h).astype(np.float64)
        x_max = (cx_np + half_w).astype(np.float64)
        y_max = (cy_np + half_h).astype(np.float64)

        gh, gw = self.grid_height, self.grid_width
        gr, gc = self.grid_row, self.grid_col

        bl_row = np.clip(np.floor(y_min / gh).astype(np.int64), 0, gr - 1)
        ur_row = np.clip(np.floor(y_max / gh).astype(np.int64), 0, gr - 1)
        bl_col = np.clip(np.floor(x_min / gw).astype(np.int64), 0, gc - 1)
        ur_col = np.clip(np.floor(x_max / gw).astype(np.int64), 0, gc - 1)

        n_rows_max = int(math.ceil(h / gh) + 2)
        n_cols_max = int(math.ceil(w / gw) + 2)

        drs = np.arange(n_rows_max, dtype=np.int64)[None, :, None]   # (1, R, 1)
        dcs = np.arange(n_cols_max, dtype=np.int64)[None, None, :]   # (1, 1, C)

        cell_rows = bl_row[:, None, None] + drs       # (B, R, C)
        cell_cols = bl_col[:, None, None] + dcs       # (B, R, C)
        valid = (cell_rows <= ur_row[:, None, None]) & (cell_cols <= ur_col[:, None, None])

        cell_x_min = cell_cols * gw
        cell_y_min = cell_rows * gh
        cell_x_max = cell_x_min + gw
        cell_y_max = cell_y_min + gh
        ox = np.maximum(0.0, np.minimum(x_max[:, None, None], cell_x_max) -
                              np.maximum(x_min[:, None, None], cell_x_min))
        oy = np.maximum(0.0, np.minimum(y_max[:, None, None], cell_y_max) -
                              np.maximum(y_min[:, None, None], cell_y_min))
        areas = (ox * oy).astype(np.float32) * valid.astype(np.float32)
        flat_idx = (cell_rows * gc + cell_cols).reshape(B, -1)
        return flat_idx, areas.reshape(B, -1), valid.reshape(B, -1)

    def _cell_areas(self, cx, cy, w, h):
        """Return (cell_indices, areas) for a macro centered at (cx,cy) with
        size (w,h). Mirrors IncrementalEvaluator._macro_density_cells."""
        half_w = w / 2.0
        half_h = h / 2.0
        x_min = cx - half_w
        y_min = cy - half_h
        x_max = cx + half_w
        y_max = cy + half_h

        bl_row = max(0, min(int(math.floor(y_min / self.grid_height)), self.grid_row - 1))
        ur_row = max(0, min(int(math.floor(y_max / self.grid_height)), self.grid_row - 1))
        bl_col = max(0, min(int(math.floor(x_min / self.grid_width)), self.grid_col - 1))
        ur_col = max(0, min(int(math.floor(x_max / self.grid_width)), self.grid_col - 1))

        cells = []
        areas = []
        for r in range(bl_row, ur_row + 1):
            for c in range(bl_col, ur_col + 1):
                cell_x_min = c * self.grid_width
                cell_y_min = r * self.grid_height
                cell_x_max = cell_x_min + self.grid_width
                cell_y_max = cell_y_min + self.grid_height
                ox = max(0.0, min(x_max, cell_x_max) - max(x_min, cell_x_min))
                oy = max(0.0, min(y_max, cell_y_max) - max(y_min, cell_y_min))
                area = ox * oy
                if area > 0:
                    cells.append(r * self.grid_col + c)
                    areas.append(area)
        return cells, areas

    # ------------------------------------------------------------------------
    # Congestion: approximate, frozen-routing + delta-blockage
    # ------------------------------------------------------------------------

    def _congestion_cost_batch(self, macro_idx, cx, cy):
        """Approximate congestion cost for B candidate positions of macro m.

        Vectorized: per-candidate macro-blockage entries are computed via
        broadcasted numpy (no Python per-candidate loop). Holds the smoothed
        per-net routing (V/H_routing_smooth) frozen at current state, and
        updates only the V/H_macro_raw blockage contribution per candidate.
        """
        w = float(self.incr.macro_w[macro_idx])
        h = float(self.incr.macro_h[macro_idx])
        if w <= 0.0 or h <= 0.0:
            return mx.full((cx.shape[0],), float(self.incr.congestion_cost), dtype=mx.float32)

        cx_np = np.asarray(cx, dtype=np.float32)
        cy_np = np.asarray(cy, dtype=np.float32)
        B = cx_np.shape[0]

        # Old blockage at current macro pos (subtract from base)
        old_x = float(self.incr.macro_pos[macro_idx, 0])
        old_y = float(self.incr.macro_pos[macro_idx, 1])
        old_entries = self._blockage_entries(old_x, old_y, w, h)
        V_base = self._V_macro_raw_np.copy()
        H_base = self._H_macro_raw_np.copy()
        for flat, v_amt, h_amt in old_entries:
            V_base[flat] -= v_amt
            H_base[flat] -= h_amt

        # Vectorized per-candidate blockage entries
        flat_idx, v_vals, h_vals, valid = self._blockage_entries_batch(cx_np, cy_np, w, h)
        # All shape (B, K_total)

        V_full = np.broadcast_to(V_base.astype(np.float32), (B, self.n_cells)).copy()
        H_full = np.broadcast_to(H_base.astype(np.float32), (B, self.n_cells)).copy()
        flat_b = np.repeat(np.arange(B), flat_idx.shape[1]).reshape(flat_idx.shape)
        flat_idx_safe = np.where(valid, flat_idx, 0)
        flat_indices = (flat_b * self.n_cells + flat_idx_safe).ravel()
        np.add.at(V_full.ravel(), flat_indices,
                  np.where(valid, v_vals, np.float32(0.0)).ravel())
        np.add.at(H_full.ravel(), flat_indices,
                  np.where(valid, h_vals, np.float32(0.0)).ravel())

        V_full_mx = mx.array(V_full)
        H_full_mx = mx.array(H_full)
        V_total = self._V_smooth_mx[None, :] + V_full_mx / np.float32(self.grid_v_routes)
        H_total = self._H_smooth_mx[None, :] + H_full_mx / np.float32(self.grid_h_routes)
        combined = mx.concatenate([V_total, H_total], axis=1)
        cnt = self.cong_top_cnt
        sorted_c = mx.sort(combined, axis=1)
        top = sorted_c[:, -cnt:]
        cong_cost = mx.sum(top, axis=1) / np.float32(cnt)
        return cong_cost

    def _blockage_entries_batch(self, cx_np, cy_np, w, h):
        """Vectorized port of `_blockage_entries` over B candidates.

        Returns flat_idx, v_vals, h_vals, valid each (B, K_total) where
        K_total = n_rows_max*n_cols_max  (main grid)
                + n_cols_max              (partial-V correction at ur_row)
                + n_rows_max              (partial-H correction at ur_col)

        Mirrors the partial-overlap correction of the verbatim CPU port.
        """
        gh, gw = self.grid_height, self.grid_width
        gr, gc = self.grid_row, self.grid_col
        vr_alloc = np.float32(self.vrouting_alloc)
        hr_alloc = np.float32(self.hrouting_alloc)

        cx32 = cx_np.astype(np.float32)
        cy32 = cy_np.astype(np.float32)
        B = cx32.shape[0]
        bl_x = (cx32 - (w / 2.0)).astype(np.float64)
        bl_y = (cy32 - (h / 2.0)).astype(np.float64)
        ur_x = (cx32 + (w / 2.0)).astype(np.float64)
        ur_y = (cy32 + (h / 2.0)).astype(np.float64)

        bl_row = np.clip(np.floor(bl_y / gh).astype(np.int64), 0, gr - 1)
        ur_row = np.clip(np.floor(ur_y / gh).astype(np.int64), 0, gr - 1)
        bl_col = np.clip(np.floor(bl_x / gw).astype(np.int64), 0, gc - 1)
        ur_col = np.clip(np.floor(ur_x / gw).astype(np.int64), 0, gc - 1)

        n_rows_max = int(math.ceil(h / gh) + 2)
        n_cols_max = int(math.ceil(w / gw) + 2)
        drs = np.arange(n_rows_max, dtype=np.int64)[None, :, None]   # (1, R, 1)
        dcs = np.arange(n_cols_max, dtype=np.int64)[None, None, :]   # (1, 1, C)

        cell_rows = bl_row[:, None, None] + drs       # (B, R, C)
        cell_cols = bl_col[:, None, None] + dcs       # (B, R, C)
        valid_main = (cell_rows <= ur_row[:, None, None]) & (cell_cols <= ur_col[:, None, None])

        cell_x_min = cell_cols * gw
        cell_y_min = cell_rows * gh
        cell_x_max = cell_x_min + gw
        cell_y_max = cell_y_min + gh
        x_dist = np.maximum(0.0, np.minimum(ur_x[:, None, None], cell_x_max) -
                                 np.maximum(bl_x[:, None, None], cell_x_min))
        y_dist = np.maximum(0.0, np.minimum(ur_y[:, None, None], cell_y_max) -
                                 np.maximum(bl_y[:, None, None], cell_y_min))
        v_amt = (x_dist * vr_alloc).astype(np.float32)
        h_amt = (y_dist * hr_alloc).astype(np.float32)
        flat_main = (cell_rows * gc + cell_cols).reshape(B, -1)
        v_main = (v_amt * valid_main.astype(np.float32)).reshape(B, -1)
        h_main = (h_amt * valid_main.astype(np.float32)).reshape(B, -1)
        valid_main_flat = valid_main.reshape(B, -1)

        # x_dist has shape (B, 1, C) (independent of row index in the macro
        # footprint) and y_dist has shape (B, R, 1). Squeeze out the
        # broadcastable axis so we have a clean (B, C) and (B, R) form.
        x_dist_2d = x_dist[:, 0, :]   # (B, C) — varies across cols
        y_dist_2d = y_dist[:, :, 0]   # (B, R) — varies across rows

        diff_y = ur_row - bl_row
        ydist_bl = y_dist_2d[:, 0]
        diff_y_clamped = np.minimum(diff_y, n_rows_max - 1)
        ydist_ur = y_dist_2d[np.arange(B), diff_y_clamped]
        partial_v = (diff_y > 0) & (
            (np.abs(ydist_bl - gh) > 1e-5) | (np.abs(ydist_ur - gh) > 1e-5))

        diff_x = ur_col - bl_col
        xdist_bl = x_dist_2d[:, 0]
        diff_x_clamped = np.minimum(diff_x, n_cols_max - 1)
        xdist_ur = x_dist_2d[np.arange(B), diff_x_clamped]
        partial_h = (diff_x > 0) & (
            (np.abs(xdist_bl - gw) > 1e-5) | (np.abs(xdist_ur - gw) > 1e-5))

        # ---- Partial-V correction at r_i = ur_row, c_i in [bl_col, ur_col] ----
        c_idx_v = bl_col[:, None] + np.arange(n_cols_max, dtype=np.int64)[None, :]  # (B, C)
        valid_v = (c_idx_v <= ur_col[:, None]) & partial_v[:, None]
        # x_dist doesn't depend on row index, so x_dist at (ur_row, c) =
        # x_dist_2d[:, c_index].
        v_sub = -(x_dist_2d.astype(np.float32) * vr_alloc)   # (B, C)
        flat_v_corr = (diff_y_clamped[:, None] + bl_row[:, None]) * gc + c_idx_v
        v_corr_v = (v_sub * valid_v.astype(np.float32))
        h_corr_v = np.zeros_like(v_corr_v)
        valid_v_flat = valid_v

        # ---- Partial-H correction at c_i = ur_col, r_i in [bl_row, ur_row] ----
        r_idx_h = bl_row[:, None] + np.arange(n_rows_max, dtype=np.int64)[None, :]  # (B, R)
        valid_h = (r_idx_h <= ur_row[:, None]) & partial_h[:, None]
        h_sub = -(y_dist_2d.astype(np.float32) * hr_alloc)  # (B, R)
        flat_h_corr = r_idx_h * gc + (diff_x_clamped[:, None] + bl_col[:, None])
        v_corr_h = np.zeros_like(h_sub)
        h_corr_h = (h_sub * valid_h.astype(np.float32))
        valid_h_flat = valid_h

        # Concatenate main + V-correction + H-correction
        flat_idx_total = np.concatenate([flat_main, flat_v_corr, flat_h_corr], axis=1)
        v_total = np.concatenate([v_main, v_corr_v, v_corr_h], axis=1)
        h_total = np.concatenate([h_main, h_corr_v, h_corr_h], axis=1)
        valid_total = np.concatenate([valid_main_flat, valid_v_flat, valid_h_flat], axis=1)
        return flat_idx_total, v_total, h_total, valid_total

    def _blockage_entries(self, cx, cy, w, h):
        """Return [(flat, v_amt, h_amt)] entries for macro at (cx,cy), size
        (w,h). Mirrors IncrementalEvaluator.__macro_route_over_grid_cell."""
        cx32 = np.float32(cx)
        cy32 = np.float32(cy)
        ur_x = cx32 + (w / 2)
        ur_y = cy32 + (h / 2)
        bl_x = cx32 - (w / 2)
        bl_y = cy32 - (h / 2)

        bl_row = max(0, min(int(math.floor(bl_y / self.grid_height)), self.grid_row - 1))
        ur_row = max(0, min(int(math.floor(ur_y / self.grid_height)), self.grid_row - 1))
        bl_col = max(0, min(int(math.floor(bl_x / self.grid_width)), self.grid_col - 1))
        ur_col = max(0, min(int(math.floor(ur_x / self.grid_width)), self.grid_col - 1))

        if_partial_v = False
        if_partial_h = False
        entries = []

        for r_i in range(bl_row, ur_row + 1):
            for c_i in range(bl_col, ur_col + 1):
                gx0 = c_i * self.grid_width
                gy0 = r_i * self.grid_height
                gx1 = gx0 + self.grid_width
                gy1 = gy0 + self.grid_height
                x_dist = max(0.0, min(ur_x, gx1) - max(bl_x, gx0))
                y_dist = max(0.0, min(ur_y, gy1) - max(bl_y, gy0))
                if ur_row != bl_row:
                    if (r_i == bl_row and abs(y_dist - self.grid_height) > 1e-5) or \
                       (r_i == ur_row and abs(y_dist - self.grid_height) > 1e-5):
                        if_partial_v = True
                if ur_col != bl_col:
                    if (c_i == bl_col and abs(x_dist - self.grid_width) > 1e-5) or \
                       (c_i == ur_col and abs(x_dist - self.grid_width) > 1e-5):
                        if_partial_h = True
                flat = r_i * self.grid_col + c_i
                v_amt = float(x_dist) * self.vrouting_alloc
                h_amt = float(y_dist) * self.hrouting_alloc
                entries.append((flat, v_amt, h_amt))

        if if_partial_v:
            for c_i in range(bl_col, ur_col + 1):
                r_i = ur_row
                gx0 = c_i * self.grid_width
                gy0 = r_i * self.grid_height
                gx1 = gx0 + self.grid_width
                gy1 = gy0 + self.grid_height
                x_dist = max(0.0, min(ur_x, gx1) - max(bl_x, gx0))
                flat = r_i * self.grid_col + c_i
                v_sub = float(x_dist) * self.vrouting_alloc
                entries.append((flat, -v_sub, 0.0))

        if if_partial_h:
            for r_i in range(bl_row, ur_row + 1):
                c_i = ur_col
                gx0 = c_i * self.grid_width
                gy0 = r_i * self.grid_height
                gx1 = gx0 + self.grid_width
                gy1 = gy0 + self.grid_height
                y_dist = max(0.0, min(ur_y, gy1) - max(bl_y, gy0))
                flat = r_i * self.grid_col + c_i
                h_sub = float(y_dist) * self.hrouting_alloc
                entries.append((flat, 0.0, -h_sub))

        return entries

    # ------------------------------------------------------------------------
    # State sync: call after `incr.move_macro(...)` is committed
    # ------------------------------------------------------------------------

    def notify_committed_move(self, macro_idx: int):
        """Refresh per-(macro, net) extremes for nets touching `macro_idx`,
        the per-cell density, and the macro/routing state. Read fresh values
        from the underlying IncrementalEvaluator."""
        incr = self.incr
        # Refresh per-(m,n) extremes for ALL macros that share a net with
        # macro_idx (their other-pin extremes shift). For macro_idx itself
        # we need to refresh extremes vs all its nets.
        affected_macros = {macro_idx}
        for nid in incr.macro_nets[macro_idx]:
            for m2 in incr.net_macros[nid]:
                affected_macros.add(int(m2))
        for m in affected_macros:
            self._refresh_macro_net_extremes(m)

        # Sync grid density
        self._grid_density_np = np.asarray(incr.grid_density, dtype=np.float64).copy()
        self._grid_density_mx = mx.array(self._grid_density_np.astype(np.float32))

        # Sync macro routing arrays
        self._V_macro_raw_np = np.asarray(incr.V_macro_raw, dtype=np.float64).copy()
        self._H_macro_raw_np = np.asarray(incr.H_macro_raw, dtype=np.float64).copy()
        self._V_macro_mx = mx.array(self._V_macro_raw_np.astype(np.float32))
        self._H_macro_mx = mx.array(self._H_macro_raw_np.astype(np.float32))

        # Smoothed routing — depends on per-net routing which can change for
        # nets touching macro_idx via the rerouting path. Refresh fully.
        self._V_smooth_np = np.asarray(incr.V_routing_smooth, dtype=np.float64).copy()
        self._H_smooth_np = np.asarray(incr.H_routing_smooth, dtype=np.float64).copy()
        self._V_smooth_mx = mx.array(self._V_smooth_np.astype(np.float32))
        self._H_smooth_mx = mx.array(self._H_smooth_np.astype(np.float32))

        # Sync totals
        self.total_hpwl = float(incr.total_hpwl)

    def notify_full_resync(self):
        """Re-pull all snapshots. Use after a sync_positions() on the CPU
        evaluator changes the placement non-incrementally."""
        self._build_macro_net_extremes()
        self._grid_density_np = np.asarray(self.incr.grid_density, dtype=np.float64).copy()
        self._V_macro_raw_np = np.asarray(self.incr.V_macro_raw, dtype=np.float64).copy()
        self._H_macro_raw_np = np.asarray(self.incr.H_macro_raw, dtype=np.float64).copy()
        self._V_smooth_np = np.asarray(self.incr.V_routing_smooth, dtype=np.float64).copy()
        self._H_smooth_np = np.asarray(self.incr.H_routing_smooth, dtype=np.float64).copy()
        self._grid_density_mx = mx.array(self._grid_density_np.astype(np.float32))
        self._V_macro_mx = mx.array(self._V_macro_raw_np.astype(np.float32))
        self._H_macro_mx = mx.array(self._H_macro_raw_np.astype(np.float32))
        self._V_smooth_mx = mx.array(self._V_smooth_np.astype(np.float32))
        self._H_smooth_mx = mx.array(self._H_smooth_np.astype(np.float32))
        self.total_hpwl = float(self.incr.total_hpwl)

    def _refresh_macro_net_extremes(self, m: int):
        """Recompute (other_x*,xoff_*,old_hpwl) for macro m's slice in the
        per-(m, net) CSR. Pin positions and net hpwls are read from incr."""
        incr = self.incr
        s = int(self.mn_starts[m])
        e = int(self.mn_starts[m + 1])
        if e == s:
            return
        pin_x = np.asarray(incr.pin_x, dtype=np.float64)
        pin_y = np.asarray(incr.pin_y, dtype=np.float64)
        pin_macro = np.asarray(incr.pin_macro, dtype=np.int32)
        pin_xoff = np.asarray(incr.pin_xoff, dtype=np.float64)
        pin_yoff = np.asarray(incr.pin_yoff, dtype=np.float64)
        net_starts = np.asarray(incr.net_starts, dtype=np.int32)
        net_hpwl = np.asarray(incr.net_hpwl, dtype=np.float64)
        # Iterate the CSR slice in the same order as build → mn_net_ids[s:e]
        for k, nid_int in enumerate(self.mn_net_ids[s:e]):
            nid = int(nid_int)
            ns = int(net_starts[nid])
            ne = int(net_starts[nid + 1])
            pm = pin_macro[ns:ne]
            mine = (pm == m)
            them = ~mine
            if them.any():
                oxs = pin_x[ns:ne][them]
                oys = pin_y[ns:ne][them]
                self._mn_other_xlo_np[s + k] = float(oxs.min())
                self._mn_other_xhi_np[s + k] = float(oxs.max())
                self._mn_other_ylo_np[s + k] = float(oys.min())
                self._mn_other_yhi_np[s + k] = float(oys.max())
            else:
                xs = pin_x[ns:ne]
                ys = pin_y[ns:ne]
                self._mn_other_xlo_np[s + k] = float(xs.min())
                self._mn_other_xhi_np[s + k] = float(xs.max())
                self._mn_other_ylo_np[s + k] = float(ys.min())
                self._mn_other_yhi_np[s + k] = float(ys.max())
            if mine.any():
                mxoff = pin_xoff[ns:ne][mine]
                myoff = pin_yoff[ns:ne][mine]
                self._mn_xoff_lo_np[s + k] = float(mxoff.min())
                self._mn_xoff_hi_np[s + k] = float(mxoff.max())
                self._mn_yoff_lo_np[s + k] = float(myoff.min())
                self._mn_yoff_hi_np[s + k] = float(myoff.max())
            self._mn_old_hpwl_np[s + k] = float(net_hpwl[nid])
