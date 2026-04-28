"""Torch-based GPU batch evaluator (backend-agnostic).

Auto-selects ``cuda`` (grader: NVIDIA RTX 6000 Ada per COMPETITION.md) >
``mps`` (M-series Macs) > ``cpu``. Same code path on every platform.

Supersedes ``_mlx_eval.py`` — MLX is Apple-Silicon-only and cannot run on
the grader, so any GPU contribution under MLX would silently fall back to
CPU at submission time. Torch satisfies both targets with one codebase.

Provides two scoring entry points:

- ``score_candidates(macro_idx, candidate_xy)`` — B candidates for one
  macro. Same semantics as the prior MLX evaluator.
- ``score_candidates_multimacro(macro_ids, candidate_xy)`` — B candidates
  spanning *multiple* macros in a single GPU dispatch. The HPWL part uses
  flat-CSR ragged batching (one entry per (candidate, net touching that
  candidate's macro), scatter-summed back). The density and congestion
  parts use a max-tile padded approach (allocate a tile big enough for the
  largest macro's footprint; mask out cells that don't actually belong to a
  given candidate's smaller macro).

The CPU IncrementalEvaluator is the source of truth for accept-or-reject;
this class only ranks. Any accepted move must also improve the CPU-exact
proxy cost.

Equivalence vs PlacementCost (matches the MLX evaluator's bounds):
- HPWL: machine-precision exact (~1e-7 max abs over 60 random moves)
- Density: machine-precision exact (~1e-7 max abs over 30 random moves)
- Total proxy: ~6e-3 max abs (frozen-routing congestion approximation)

The CPU validator on commit ensures the approximation never poisons the
placement.
"""
from __future__ import annotations
import math
import numpy as np
import torch


def _select_device():
    """Auto-select the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchBatchEvaluator:
    """Backend-agnostic batch evaluator. ``backend.device`` is auto-selected.

    Construct with a sync'd ``IncrementalEvaluator``. Call
    ``notify_committed_move(macro_idx)`` after every CPU-accepted move to
    refresh the affected per-(macro, net) extremes and the per-cell grids.
    Call ``notify_full_resync()`` after a non-incremental ``sync_positions``.
    """

    def __init__(self, incr_eval, benchmark, *, device=None, dtype=torch.float32):
        self.device = device or _select_device()
        self.dtype = dtype
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
        # Top 5% over (V_total ∪ H_total) — matches PlacementCost.
        self.cong_top_cnt = max(1, math.floor(2 * self.n_cells * 0.05))

        # Per-(macro, net) extremes (CSR) — initially in numpy; mirrored to
        # torch tensors on the device.
        self._build_macro_net_extremes_np()
        self._upload_macro_net_extremes()

        # Macro-size tensor (per-macro w, h, used for density/congestion
        # tile sizing).
        self._macro_w_np = np.asarray(incr_eval.macro_w, dtype=np.float64)
        self._macro_h_np = np.asarray(incr_eval.macro_h, dtype=np.float64)
        # Movable hard-macro maximum size — used to size the padded tile in
        # `score_candidates_multimacro` (max footprint over all hard macros
        # we might evaluate).
        self._max_w_hard = float(self._macro_w_np[:self.n_hard].max())
        self._max_h_hard = float(self._macro_h_np[:self.n_hard].max())

        # Static grid state mirrored to device.
        self._grid_density_np = np.asarray(incr_eval.grid_density, dtype=np.float64).copy()
        self._V_macro_np = np.asarray(incr_eval.V_macro_raw, dtype=np.float64).copy()
        self._H_macro_np = np.asarray(incr_eval.H_macro_raw, dtype=np.float64).copy()
        self._V_smooth_np = np.asarray(incr_eval.V_routing_smooth, dtype=np.float64).copy()
        self._H_smooth_np = np.asarray(incr_eval.H_routing_smooth, dtype=np.float64).copy()
        self._upload_grids()

        # Current totals (scalars; updated lazily from incr).
        self.total_hpwl = float(incr_eval.total_hpwl)
        self._wl_norm = (self.cw + self.ch) * self.net_cnt_value

    # ------------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------------

    def _build_macro_net_extremes_np(self):
        """Mirrors `_mlx_eval.MLXBatchEvaluator._build_macro_net_extremes` but
        keeps results on numpy until upload."""
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
                    xs = pin_x[s:e]
                    ys = pin_y[s:e]
                    other_xlo.append(float(xs.min()))
                    other_xhi.append(float(xs.max()))
                    other_ylo.append(float(ys.min()))
                    other_yhi.append(float(ys.max()))

                if mine.any():
                    mxoff = pin_xoff[s:e][mine]
                    myoff = pin_yoff[s:e][mine]
                    xoff_lo.append(float(mxoff.min()))
                    xoff_hi.append(float(mxoff.max()))
                    yoff_lo.append(float(myoff.min()))
                    yoff_hi.append(float(myoff.max()))
                else:
                    xoff_lo.append(0.0)
                    xoff_hi.append(0.0)
                    yoff_lo.append(0.0)
                    yoff_hi.append(0.0)

                net_ids.append(int(nid))
                weights.append(float(net_weight[nid]))
                old_hpwl.append(float(net_hpwl[nid]))
            starts.append(len(net_ids))

        self.mn_starts_np = np.asarray(starts, dtype=np.int64)
        self.mn_net_ids_np = np.asarray(net_ids, dtype=np.int64)
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

    def _upload_macro_net_extremes(self):
        d = self.device
        # Integer indices stay int64 (torch's default for indexing).
        self.mn_starts = torch.as_tensor(self.mn_starts_np, dtype=torch.int64, device=d)
        # Per-(m, net) numerical fields → float32 on device for compute speed.
        # Match the MLX evaluator's float32 cast precision.
        self._mn_other_xlo = torch.as_tensor(self._mn_other_xlo_np.astype(np.float32), device=d)
        self._mn_other_xhi = torch.as_tensor(self._mn_other_xhi_np.astype(np.float32), device=d)
        self._mn_other_ylo = torch.as_tensor(self._mn_other_ylo_np.astype(np.float32), device=d)
        self._mn_other_yhi = torch.as_tensor(self._mn_other_yhi_np.astype(np.float32), device=d)
        self._mn_xoff_lo = torch.as_tensor(self._mn_xoff_lo_np.astype(np.float32), device=d)
        self._mn_xoff_hi = torch.as_tensor(self._mn_xoff_hi_np.astype(np.float32), device=d)
        self._mn_yoff_lo = torch.as_tensor(self._mn_yoff_lo_np.astype(np.float32), device=d)
        self._mn_yoff_hi = torch.as_tensor(self._mn_yoff_hi_np.astype(np.float32), device=d)
        self._mn_weight = torch.as_tensor(self._mn_weight_np.astype(np.float32), device=d)
        self._mn_old_hpwl = torch.as_tensor(self._mn_old_hpwl_np.astype(np.float32), device=d)

    def _upload_grids(self):
        d = self.device
        self._grid_density_t = torch.as_tensor(self._grid_density_np.astype(np.float32), device=d)
        self._V_macro_t = torch.as_tensor(self._V_macro_np.astype(np.float32), device=d)
        self._H_macro_t = torch.as_tensor(self._H_macro_np.astype(np.float32), device=d)
        self._V_smooth_t = torch.as_tensor(self._V_smooth_np.astype(np.float32), device=d)
        self._H_smooth_t = torch.as_tensor(self._H_smooth_np.astype(np.float32), device=d)

    # ------------------------------------------------------------------------
    # Public API: single-macro convenience (delegates to multimacro)
    # ------------------------------------------------------------------------

    def score_candidates(self, macro_idx, candidate_xy,
                         skip_density=False, skip_congestion=False):
        """B candidates for one macro. Returns (B,) torch.float32 on device."""
        cand_t = self._as_device_tensor(candidate_xy)
        B = int(cand_t.shape[0])
        macro_ids = torch.full((B,), int(macro_idx), dtype=torch.int64, device=self.device)
        wl, dens, cong = self._score_components_multimacro(
            macro_ids, cand_t, skip_density, skip_congestion)
        return wl + 0.5 * dens + 0.5 * cong

    def score_components(self, macro_idx, candidate_xy,
                         skip_density=False, skip_congestion=False):
        """Like score_candidates but returns (wl, density, congestion) tuple."""
        cand_t = self._as_device_tensor(candidate_xy)
        B = int(cand_t.shape[0])
        macro_ids = torch.full((B,), int(macro_idx), dtype=torch.int64, device=self.device)
        return self._score_components_multimacro(
            macro_ids, cand_t, skip_density, skip_congestion)

    def score_candidates_multimacro(self, macro_ids, candidate_xy,
                                    skip_density=False, skip_congestion=False):
        """Score B candidates each for an arbitrary macro.

        Parameters
        ----------
        macro_ids : (B,) int (numpy or torch)
            macro index per candidate. May contain duplicates.
        candidate_xy : (B, 2) float (numpy or torch)
        skip_density, skip_congestion : bool
            See `score_candidates`.

        Returns
        -------
        scores : (B,) torch.float32 on device
            proxy_cost = wirelength + 0.5*density + 0.5*congestion
        """
        m_t = self._as_device_tensor(macro_ids, dtype=torch.int64)
        c_t = self._as_device_tensor(candidate_xy)
        wl, dens, cong = self._score_components_multimacro(
            m_t, c_t, skip_density, skip_congestion)
        return wl + 0.5 * dens + 0.5 * cong

    def _score_components_multimacro(self, macro_ids, candidate_xy,
                                     skip_density, skip_congestion):
        wl = self._wl_cost_multimacro(macro_ids, candidate_xy)
        if skip_density:
            B = int(macro_ids.shape[0])
            dens = torch.zeros(B, dtype=torch.float32, device=self.device)
        else:
            dens = self._density_cost_multimacro(macro_ids, candidate_xy)
        if skip_congestion:
            B = int(macro_ids.shape[0])
            cong = torch.full((B,), float(self.incr.congestion_cost),
                              dtype=torch.float32, device=self.device)
        else:
            cong = self._congestion_cost_multimacro(macro_ids, candidate_xy)
        return wl, dens, cong

    # ------------------------------------------------------------------------
    # HPWL via flat-CSR ragged batching
    # ------------------------------------------------------------------------

    def _wl_cost_multimacro(self, macro_ids, candidate_xy):
        """Compute per-candidate wirelength_cost.

        Flat-CSR scheme: for each candidate b, expand into one row per net
        touching that candidate's macro. Compute per-row HPWL delta, then
        scatter-sum back to (B,) by candidate index.
        """
        B = int(macro_ids.shape[0])
        # Use CPU-side numpy to compute the flat ragged layout (small, no
        # GPU win to be had on this prep).
        m_np = macro_ids.detach().cpu().numpy()
        starts = self.mn_starts_np[m_np]
        ends = self.mn_starts_np[m_np + 1]
        counts = (ends - starts).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            wl_total = torch.full((B,), float(self.total_hpwl),
                                  dtype=torch.float32, device=self.device)
            return wl_total / float(self._wl_norm)

        # Flat indices into the CSR for each entry.
        offsets = np.zeros(B + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        # b_per_entry[i] = which candidate this entry belongs to
        b_per_entry_np = np.repeat(np.arange(B, dtype=np.int64), counts)
        # csr_row[i] = starts[b_per_entry[i]] + (i - offsets[b_per_entry[i]])
        within = np.arange(total, dtype=np.int64) - offsets[b_per_entry_np]
        csr_row_np = starts[b_per_entry_np] + within

        d = self.device
        b_per_entry = torch.as_tensor(b_per_entry_np, device=d)
        csr_row = torch.as_tensor(csr_row_np, device=d)

        # Gather per-(m, n) extremes for each entry.
        other_xlo = self._mn_other_xlo[csr_row]
        other_xhi = self._mn_other_xhi[csr_row]
        other_ylo = self._mn_other_ylo[csr_row]
        other_yhi = self._mn_other_yhi[csr_row]
        xoff_lo = self._mn_xoff_lo[csr_row]
        xoff_hi = self._mn_xoff_hi[csr_row]
        yoff_lo = self._mn_yoff_lo[csr_row]
        yoff_hi = self._mn_yoff_hi[csr_row]
        weight = self._mn_weight[csr_row]
        old_hpwl = self._mn_old_hpwl[csr_row]

        # Gather (cx, cy) per entry from candidate_xy by b_per_entry.
        cx_t = candidate_xy[:, 0]
        cy_t = candidate_xy[:, 1]
        cx_per_entry = cx_t[b_per_entry]
        cy_per_entry = cy_t[b_per_entry]

        new_xlo = torch.minimum(other_xlo, cx_per_entry + xoff_lo)
        new_xhi = torch.maximum(other_xhi, cx_per_entry + xoff_hi)
        new_ylo = torch.minimum(other_ylo, cy_per_entry + yoff_lo)
        new_yhi = torch.maximum(other_yhi, cy_per_entry + yoff_hi)
        new_hpwl = (new_xhi - new_xlo) + (new_yhi - new_ylo)
        delta_per_entry = weight * (new_hpwl - old_hpwl)

        # Scatter-sum to per-candidate.
        delta_per_b = torch.zeros(B, dtype=torch.float32, device=d)
        delta_per_b.index_add_(0, b_per_entry, delta_per_entry)

        wl_total = float(self.total_hpwl) + delta_per_b
        return wl_total / float(self._wl_norm)

    # ------------------------------------------------------------------------
    # Density via max-tile padded approach
    # ------------------------------------------------------------------------

    def _density_cost_multimacro(self, macro_ids, candidate_xy):
        """Compute per-candidate density_cost. Each candidate b has macro
        m=macro_ids[b], size (w_m, h_m). Allocate a tile big enough for the
        largest macro in macro_ids, mask cells outside the actual footprint.

        Implementation:
          base[c] = current grid_density[c] - (sum over candidates of old
                    contribution) — but candidates with the same macro share
                    old contribution; different macros have different olds.
          For correctness across mixed-macro batches: we compute each
          candidate's full new grid as
            new[b, c] = current[c] - delta_old_b[c] + delta_new_b[c]
          where delta_old_b is the contribution of macro_ids[b] at its
          *current* position (snapshot at construction), and delta_new_b is
          the contribution at candidate_xy[b].
        """
        B = int(macro_ids.shape[0])
        d = self.device

        m_np = macro_ids.detach().cpu().numpy()
        c_np = candidate_xy.detach().cpu().numpy().astype(np.float32)

        w_per = self._macro_w_np[m_np]
        h_per = self._macro_h_np[m_np]
        zero_mask = (w_per <= 0.0) | (h_per <= 0.0)
        # Replace with safe values for computation; we'll restore the
        # current density_cost for those at the end.
        if zero_mask.any():
            w_per = np.where(zero_mask, 1e-6, w_per)
            h_per = np.where(zero_mask, 1e-6, h_per)

        # Max tile size — pad to the largest macro in this batch.
        max_w = float(w_per.max())
        max_h = float(h_per.max())
        n_rows_max = int(math.ceil(max_h / self.grid_height) + 2)
        n_cols_max = int(math.ceil(max_w / self.grid_width) + 2)

        # New (per-candidate) footprint cells & areas in the max tile.
        new_flat_idx, new_areas, new_valid = self._cell_areas_batch_multi(
            c_np[:, 0], c_np[:, 1], w_per, h_per, n_rows_max, n_cols_max)
        # Old (per-macro current pos) — same vector size for symmetry.
        cur_pos = np.asarray(self.incr.macro_pos, dtype=np.float32)
        old_cx = cur_pos[m_np, 0]
        old_cy = cur_pos[m_np, 1]
        old_flat_idx, old_areas, old_valid = self._cell_areas_batch_multi(
            old_cx, old_cy, w_per, h_per, n_rows_max, n_cols_max)

        # Build (B, n_cells) tensor of per-candidate density.
        # base = grid_density (B copies) - delta_old_b + delta_new_b
        base = self._grid_density_np.astype(np.float32)
        full = np.broadcast_to(base, (B, self.n_cells)).copy()
        flat_b = np.repeat(np.arange(B, dtype=np.int64),
                           new_flat_idx.shape[1]).reshape(new_flat_idx.shape)
        # Subtract old
        old_idx_safe = np.where(old_valid, old_flat_idx, 0)
        old_lin = (flat_b * self.n_cells + old_idx_safe).ravel()
        old_vals = -(old_areas / np.float32(self.grid_area))
        old_vals = np.where(old_valid, old_vals, np.float32(0.0)).ravel()
        np.add.at(full.ravel(), old_lin, old_vals)
        # Add new
        new_idx_safe = np.where(new_valid, new_flat_idx, 0)
        new_lin = (flat_b * self.n_cells + new_idx_safe).ravel()
        new_vals = (new_areas / np.float32(self.grid_area))
        new_vals = np.where(new_valid, new_vals, np.float32(0.0)).ravel()
        np.add.at(full.ravel(), new_lin, new_vals)

        full_t = torch.as_tensor(full, device=d)
        cnt = self.density_cnt
        # Top-cnt sum per row.
        top_vals, _ = torch.topk(full_t, k=cnt, dim=1, largest=True, sorted=False)
        density_cost = top_vals.sum(dim=1) / float(cnt) * 0.5

        # Restore current density_cost for zero-area candidates.
        if zero_mask.any():
            current = float(self.incr.density_cost)
            zm = torch.as_tensor(zero_mask, dtype=torch.bool, device=d)
            density_cost = torch.where(zm,
                                        torch.full_like(density_cost, current),
                                        density_cost)
        return density_cost

    def _cell_areas_batch_multi(self, cx_np, cy_np, w_np, h_np,
                                n_rows_max, n_cols_max):
        """Vectorized per-candidate density-cell areas in a (B, R*C) tile.

        cx_np, cy_np : (B,) float32 — candidate centers
        w_np, h_np   : (B,) float64 — per-candidate macro size
        Returns flat_idx, areas, valid each (B, R*C).
        """
        B = cx_np.shape[0]
        gh, gw = self.grid_height, self.grid_width
        gr, gc = self.grid_row, self.grid_col

        half_w = (w_np / 2.0)
        half_h = (h_np / 2.0)
        x_min = (cx_np - half_w).astype(np.float64)
        y_min = (cy_np - half_h).astype(np.float64)
        x_max = (cx_np + half_w).astype(np.float64)
        y_max = (cy_np + half_h).astype(np.float64)

        bl_row = np.clip(np.floor(y_min / gh).astype(np.int64), 0, gr - 1)
        ur_row = np.clip(np.floor(y_max / gh).astype(np.int64), 0, gr - 1)
        bl_col = np.clip(np.floor(x_min / gw).astype(np.int64), 0, gc - 1)
        ur_col = np.clip(np.floor(x_max / gw).astype(np.int64), 0, gc - 1)

        drs = np.arange(n_rows_max, dtype=np.int64)[None, :, None]
        dcs = np.arange(n_cols_max, dtype=np.int64)[None, None, :]

        cell_rows = bl_row[:, None, None] + drs   # (B, R, 1)
        cell_cols = bl_col[:, None, None] + dcs   # (B, 1, C)
        valid = (cell_rows <= ur_row[:, None, None]) & \
                (cell_cols <= ur_col[:, None, None])

        cell_x_min = cell_cols * gw
        cell_y_min = cell_rows * gh
        cell_x_max = cell_x_min + gw
        cell_y_max = cell_y_min + gh
        ox = np.maximum(0.0,
                        np.minimum(x_max[:, None, None], cell_x_max) -
                        np.maximum(x_min[:, None, None], cell_x_min))
        oy = np.maximum(0.0,
                        np.minimum(y_max[:, None, None], cell_y_max) -
                        np.maximum(y_min[:, None, None], cell_y_min))
        areas = (ox * oy).astype(np.float32) * valid.astype(np.float32)
        flat_idx = (cell_rows * gc + cell_cols)
        # Broadcast to (B, R, C) explicitly so reshape(B, -1) works.
        flat_idx_full = np.broadcast_to(flat_idx, (B, n_rows_max, n_cols_max))
        valid_full = np.broadcast_to(valid, (B, n_rows_max, n_cols_max))
        return (flat_idx_full.reshape(B, -1).copy(),
                areas.reshape(B, -1).copy(),
                valid_full.reshape(B, -1).copy())

    # ------------------------------------------------------------------------
    # Congestion via max-tile padded approach
    # ------------------------------------------------------------------------

    def _congestion_cost_multimacro(self, macro_ids, candidate_xy):
        """Approximate congestion cost (frozen routing demand + per-candidate
        macro-blockage delta).

        Same max-tile strategy as density.
        """
        B = int(macro_ids.shape[0])
        d = self.device

        m_np = macro_ids.detach().cpu().numpy()
        c_np = candidate_xy.detach().cpu().numpy().astype(np.float32)

        w_per = self._macro_w_np[m_np]
        h_per = self._macro_h_np[m_np]
        zero_mask = (w_per <= 0.0) | (h_per <= 0.0)
        if zero_mask.any():
            w_per = np.where(zero_mask, 1e-6, w_per)
            h_per = np.where(zero_mask, 1e-6, h_per)

        max_w = float(w_per.max())
        max_h = float(h_per.max())
        n_rows_max = int(math.ceil(max_h / self.grid_height) + 2)
        n_cols_max = int(math.ceil(max_w / self.grid_width) + 2)

        # New blockage per candidate.
        new_flat, new_v, new_h, new_valid = self._blockage_entries_multi(
            c_np[:, 0], c_np[:, 1], w_per, h_per, n_rows_max, n_cols_max)
        # Old blockage at macro's current pos.
        cur_pos = np.asarray(self.incr.macro_pos, dtype=np.float32)
        old_cx = cur_pos[m_np, 0]
        old_cy = cur_pos[m_np, 1]
        old_flat, old_v, old_h, old_valid = self._blockage_entries_multi(
            old_cx, old_cy, w_per, h_per, n_rows_max, n_cols_max)

        # Build (B, n_cells) V_macro and H_macro
        V_full = np.broadcast_to(self._V_macro_np.astype(np.float32),
                                 (B, self.n_cells)).copy()
        H_full = np.broadcast_to(self._H_macro_np.astype(np.float32),
                                 (B, self.n_cells)).copy()
        flat_b = np.repeat(np.arange(B, dtype=np.int64),
                           new_flat.shape[1]).reshape(new_flat.shape)

        # Subtract old contribution
        old_idx_safe = np.where(old_valid, old_flat, 0)
        old_lin = (flat_b * self.n_cells + old_idx_safe).ravel()
        np.add.at(V_full.ravel(), old_lin,
                  np.where(old_valid, -old_v, np.float32(0.0)).ravel())
        np.add.at(H_full.ravel(), old_lin,
                  np.where(old_valid, -old_h, np.float32(0.0)).ravel())

        # Add new contribution
        new_idx_safe = np.where(new_valid, new_flat, 0)
        new_lin = (flat_b * self.n_cells + new_idx_safe).ravel()
        np.add.at(V_full.ravel(), new_lin,
                  np.where(new_valid, new_v, np.float32(0.0)).ravel())
        np.add.at(H_full.ravel(), new_lin,
                  np.where(new_valid, new_h, np.float32(0.0)).ravel())

        V_t = torch.as_tensor(V_full, device=d)
        H_t = torch.as_tensor(H_full, device=d)
        V_total = self._V_smooth_t.unsqueeze(0) + V_t / float(self.grid_v_routes)
        H_total = self._H_smooth_t.unsqueeze(0) + H_t / float(self.grid_h_routes)
        combined = torch.cat([V_total, H_total], dim=1)
        cnt = self.cong_top_cnt
        top_vals, _ = torch.topk(combined, k=cnt, dim=1, largest=True, sorted=False)
        cong_cost = top_vals.sum(dim=1) / float(cnt)

        if zero_mask.any():
            current = float(self.incr.congestion_cost)
            zm = torch.as_tensor(zero_mask, dtype=torch.bool, device=d)
            cong_cost = torch.where(zm,
                                     torch.full_like(cong_cost, current),
                                     cong_cost)
        return cong_cost

    def _blockage_entries_multi(self, cx_np, cy_np, w_np, h_np,
                                n_rows_max, n_cols_max):
        """Vectorized per-candidate macro-blockage entries (mirror of
        `_mlx_eval._blockage_entries_batch` but for multi-macro)."""
        B = cx_np.shape[0]
        gh, gw = self.grid_height, self.grid_width
        gr, gc = self.grid_row, self.grid_col
        vr_alloc = np.float32(self.vrouting_alloc)
        hr_alloc = np.float32(self.hrouting_alloc)

        cx32 = cx_np.astype(np.float32)
        cy32 = cy_np.astype(np.float32)
        bl_x = (cx32 - (w_np / 2.0)).astype(np.float64)
        bl_y = (cy32 - (h_np / 2.0)).astype(np.float64)
        ur_x = (cx32 + (w_np / 2.0)).astype(np.float64)
        ur_y = (cy32 + (h_np / 2.0)).astype(np.float64)

        bl_row = np.clip(np.floor(bl_y / gh).astype(np.int64), 0, gr - 1)
        ur_row = np.clip(np.floor(ur_y / gh).astype(np.int64), 0, gr - 1)
        bl_col = np.clip(np.floor(bl_x / gw).astype(np.int64), 0, gc - 1)
        ur_col = np.clip(np.floor(ur_x / gw).astype(np.int64), 0, gc - 1)

        drs = np.arange(n_rows_max, dtype=np.int64)[None, :, None]
        dcs = np.arange(n_cols_max, dtype=np.int64)[None, None, :]
        cell_rows = bl_row[:, None, None] + drs
        cell_cols = bl_col[:, None, None] + dcs
        valid_main = (cell_rows <= ur_row[:, None, None]) & \
                     (cell_cols <= ur_col[:, None, None])

        cell_x_min = cell_cols * gw
        cell_y_min = cell_rows * gh
        cell_x_max = cell_x_min + gw
        cell_y_max = cell_y_min + gh
        x_dist = np.maximum(0.0,
                            np.minimum(ur_x[:, None, None], cell_x_max) -
                            np.maximum(bl_x[:, None, None], cell_x_min))
        y_dist = np.maximum(0.0,
                            np.minimum(ur_y[:, None, None], cell_y_max) -
                            np.maximum(bl_y[:, None, None], cell_y_min))
        v_amt = (x_dist * vr_alloc).astype(np.float32)
        h_amt = (y_dist * hr_alloc).astype(np.float32)

        flat_main = np.broadcast_to(cell_rows * gc + cell_cols,
                                    (B, n_rows_max, n_cols_max)).reshape(B, -1).copy()
        v_main_full = np.broadcast_to(v_amt, (B, n_rows_max, n_cols_max)).copy()
        h_main_full = np.broadcast_to(h_amt, (B, n_rows_max, n_cols_max)).copy()
        valid_main_full = np.broadcast_to(valid_main, (B, n_rows_max, n_cols_max)).copy()
        v_main = (v_main_full * valid_main_full.astype(np.float32)).reshape(B, -1)
        h_main = (h_main_full * valid_main_full.astype(np.float32)).reshape(B, -1)
        valid_main_flat = valid_main_full.reshape(B, -1)

        # Partial-overlap correction
        x_dist_2d = np.broadcast_to(x_dist, (B, n_rows_max, n_cols_max))[:, 0, :]
        y_dist_2d = np.broadcast_to(y_dist, (B, n_rows_max, n_cols_max))[:, :, 0]

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

        # Partial-V correction at r_i = ur_row, c_i in [bl_col, ur_col]
        c_idx_v = bl_col[:, None] + np.arange(n_cols_max, dtype=np.int64)[None, :]
        valid_v = (c_idx_v <= ur_col[:, None]) & partial_v[:, None]
        v_sub = -(x_dist_2d.astype(np.float32) * vr_alloc)
        flat_v_corr = (diff_y_clamped[:, None] + bl_row[:, None]) * gc + c_idx_v
        v_corr_v = v_sub * valid_v.astype(np.float32)
        h_corr_v = np.zeros_like(v_corr_v)
        valid_v_flat = valid_v

        # Partial-H correction at c_i = ur_col, r_i in [bl_row, ur_row]
        r_idx_h = bl_row[:, None] + np.arange(n_rows_max, dtype=np.int64)[None, :]
        valid_h = (r_idx_h <= ur_row[:, None]) & partial_h[:, None]
        h_sub = -(y_dist_2d.astype(np.float32) * hr_alloc)
        flat_h_corr = r_idx_h * gc + (diff_x_clamped[:, None] + bl_col[:, None])
        v_corr_h = np.zeros_like(h_sub)
        h_corr_h = h_sub * valid_h.astype(np.float32)
        valid_h_flat = valid_h

        flat_total = np.concatenate([flat_main, flat_v_corr, flat_h_corr], axis=1)
        v_total = np.concatenate([v_main, v_corr_v, v_corr_h], axis=1)
        h_total = np.concatenate([h_main, h_corr_v, h_corr_h], axis=1)
        valid_total = np.concatenate([valid_main_flat, valid_v_flat, valid_h_flat], axis=1)
        return flat_total, v_total, h_total, valid_total

    # ------------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------------

    def notify_committed_move(self, macro_idx):
        """Refresh per-(m, n) extremes for nets touching `macro_idx`, and
        re-upload grid state. Mirrors `MLXBatchEvaluator.notify_committed_move`."""
        incr = self.incr
        affected = {int(macro_idx)}
        for nid in incr.macro_nets[macro_idx]:
            for m2 in incr.net_macros[nid]:
                affected.add(int(m2))
        for m in affected:
            self._refresh_macro_net_extremes_np(m)
        self._upload_macro_net_extremes()

        self._grid_density_np = np.asarray(incr.grid_density, dtype=np.float64).copy()
        self._V_macro_np = np.asarray(incr.V_macro_raw, dtype=np.float64).copy()
        self._H_macro_np = np.asarray(incr.H_macro_raw, dtype=np.float64).copy()
        self._V_smooth_np = np.asarray(incr.V_routing_smooth, dtype=np.float64).copy()
        self._H_smooth_np = np.asarray(incr.H_routing_smooth, dtype=np.float64).copy()
        self._upload_grids()
        self.total_hpwl = float(incr.total_hpwl)

    def notify_full_resync(self):
        """After a non-incremental sync_positions, re-pull all snapshots."""
        self._build_macro_net_extremes_np()
        self._upload_macro_net_extremes()
        self._grid_density_np = np.asarray(self.incr.grid_density, dtype=np.float64).copy()
        self._V_macro_np = np.asarray(self.incr.V_macro_raw, dtype=np.float64).copy()
        self._H_macro_np = np.asarray(self.incr.H_macro_raw, dtype=np.float64).copy()
        self._V_smooth_np = np.asarray(self.incr.V_routing_smooth, dtype=np.float64).copy()
        self._H_smooth_np = np.asarray(self.incr.H_routing_smooth, dtype=np.float64).copy()
        self._upload_grids()
        self.total_hpwl = float(self.incr.total_hpwl)

    def _refresh_macro_net_extremes_np(self, m):
        """Recompute (other_x*, xoff_*, old_hpwl) for macro m's CSR slice."""
        incr = self.incr
        s = int(self.mn_starts_np[m])
        e = int(self.mn_starts_np[m + 1])
        if e == s:
            return
        pin_x = np.asarray(incr.pin_x, dtype=np.float64)
        pin_y = np.asarray(incr.pin_y, dtype=np.float64)
        pin_macro = np.asarray(incr.pin_macro, dtype=np.int32)
        pin_xoff = np.asarray(incr.pin_xoff, dtype=np.float64)
        pin_yoff = np.asarray(incr.pin_yoff, dtype=np.float64)
        net_starts = np.asarray(incr.net_starts, dtype=np.int32)
        net_hpwl = np.asarray(incr.net_hpwl, dtype=np.float64)
        for k, nid_int in enumerate(self.mn_net_ids_np[s:e]):
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

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def _as_device_tensor(self, x, dtype=None):
        """Convert numpy/list/torch input to a tensor on this evaluator's
        device. Default dtype is self.dtype (float32) for floating inputs;
        int64 for integer inputs."""
        if isinstance(x, torch.Tensor):
            t = x
            if t.device != self.device:
                t = t.to(self.device)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            return t
        arr = np.asarray(x)
        if dtype is None:
            if np.issubdtype(arr.dtype, np.integer):
                dtype = torch.int64
            else:
                dtype = self.dtype
        return torch.as_tensor(arr, dtype=dtype, device=self.device)
