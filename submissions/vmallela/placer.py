"""
Optimized Macro Placer — vmallela submission

Pipeline:
  1. Push-apart pre-processing (3 configs: conservative, moderate, aggressive)
  2. Multi-start greedy legalization (20 orderings x 4 step sizes x 4 starts)
  3. Refine toward initial positions
  4. Coordinate descent with connectivity-aware ordering, size-scaled deltas,
     first-improving for large moves, best-of-all for fine-grained moves
  5. Perturbation phase: random displacement + connected-pair swaps to escape
     local optima

Usage:
    uv run evaluate submissions/vmallela/placer.py
    uv run evaluate submissions/vmallela/placer.py --all
"""

import math
import multiprocessing as mp
import random
import time
import numpy as np
import torch
from pathlib import Path
from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# PlacementCost loader
# ---------------------------------------------------------------------------

def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


# ---------------------------------------------------------------------------
# Incremental proxy cost evaluator
# ---------------------------------------------------------------------------

class IncrementalEvaluator:
    """Fast incremental proxy cost evaluator using numpy arrays.

    Mirrors PlacementCost's wirelength, density, and congestion computations
    but supports O(1ms) incremental updates when a single macro moves,
    vs O(1.3s) for the full PlacementCost recomputation.

    Usage:
        incr = IncrementalEvaluator(plc, benchmark)
        cost = incr.get_proxy_cost()
        new_cost = incr.move_macro(3, new_x, new_y)
        incr.undo_move()  # restore previous state
    """

    def __init__(self, plc, benchmark):
        n_hard = benchmark.num_hard_macros
        self.n_hard = n_hard
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)

        # Grid parameters
        self.grid_col = plc.grid_col
        self.grid_row = plc.grid_row
        self.grid_width = float(plc.width / plc.grid_col)
        self.grid_height = float(plc.height / plc.grid_row)
        self.grid_area = self.grid_width * self.grid_height
        self.n_cells = self.grid_col * self.grid_row

        # Routing parameters
        self.grid_v_routes = self.grid_width * plc.vroutes_per_micron
        self.grid_h_routes = self.grid_height * plc.hroutes_per_micron
        self.smooth_range = int(plc.smooth_range)
        self.vrouting_alloc = plc.vrouting_alloc
        self.hrouting_alloc = plc.hrouting_alloc

        # Weighted net count for wirelength normalization
        self.net_cnt = plc.net_cnt if plc.net_cnt != 0 else 1.0

        # --- Build macro data ---
        # Map plc indices to bench indices
        self._plc_to_hard = {}
        for bidx, plc_idx in enumerate(plc.hard_macro_indices):
            self._plc_to_hard[plc.modules_w_pins[plc_idx].get_name()] = bidx

        self._plc_to_soft = {}
        for bidx, plc_idx in enumerate(plc.soft_macro_indices):
            self._plc_to_soft[plc.modules_w_pins[plc_idx].get_name()] = bidx

        self._plc_to_port = {}
        for plc_idx in plc.port_indices:
            self._plc_to_port[plc.modules_w_pins[plc_idx].get_name()] = plc_idx

        # Macro sizes and positions — use float64 from plc (not float32 from benchmark tensors)
        # to avoid float32→float64 precision loss at grid cell boundaries.
        n_total = benchmark.macro_positions.shape[0]
        self.macro_w = np.zeros(n_total, dtype=np.float64)
        self.macro_h = np.zeros(n_total, dtype=np.float64)
        # float32 to match PlacementCost's internal storage (avoids grid-cell-boundary
        # mismatches due to float32↔float64 promotion differences)
        self.macro_pos = np.zeros((n_total, 2), dtype=np.float32)

        # Sizes: read from plc (float64 precision, doesn't change)
        for bidx, plc_idx in enumerate(plc.hard_macro_indices):
            mod = plc.modules_w_pins[plc_idx]
            self.macro_w[bidx] = mod.get_width()
            self.macro_h[bidx] = mod.get_height()
        for bidx, plc_idx in enumerate(plc.soft_macro_indices):
            mod = plc.modules_w_pins[plc_idx]
            self.macro_w[n_hard + bidx] = mod.get_width()
            self.macro_h[n_hard + bidx] = mod.get_height()

        # Positions: use float32→float64 converted positions matching what _set_placement does
        # (benchmark.macro_positions is float32; converting to float64 introduces precision loss
        #  at grid cell boundaries. We must match this exactly.)
        from macro_place.objective import _set_placement
        _set_placement(plc, benchmark.macro_positions, benchmark)
        for bidx, plc_idx in enumerate(plc.hard_macro_indices):
            x, y = plc.modules_w_pins[plc_idx].get_pos()
            self.macro_pos[bidx] = [x, y]
        for bidx, plc_idx in enumerate(plc.soft_macro_indices):
            x, y = plc.modules_w_pins[plc_idx].get_pos()
            self.macro_pos[n_hard + bidx] = [x, y]

        # --- Build pin/net data ---
        # Pin offset map: pin_name -> (hard_macro_bidx, x_off, y_off) or None
        pin_offsets = {}
        for plc_idx, mod in enumerate(plc.modules_w_pins):
            if mod.get_type() == 'MACRO_PIN':
                parent = mod.get_name().split("/")[0]
                if parent in self._plc_to_hard:
                    xo, yo = mod.get_offset()
                    pin_offsets[mod.get_name()] = (self._plc_to_hard[parent], xo, yo)
                elif parent in self._plc_to_soft:
                    si = self._plc_to_soft[parent]
                    pin_offsets[mod.get_name()] = (n_hard + si, 0.0, 0.0)

        # Port positions (fixed)
        port_positions = {}
        for plc_idx in plc.port_indices:
            mod = plc.modules_w_pins[plc_idx]
            port_positions[mod.get_name()] = mod.get_pos()

        # Build nets in CSR format
        net_pin_list = []  # list of lists of (macro_bidx_or_-1, x_off, y_off)
        net_weights = []
        net_drivers = []  # index into flat pin array for each net's driver
        macro_nets = [[] for _ in range(n_total)]  # macro -> [net_ids]
        flat_pins = []  # (macro_bidx_or_-1, x_off, y_off)
        net_starts = [0]

        net_id = 0
        for driver_name, sinks in plc.nets.items():
            driver_plc_idx = plc.mod_name_to_indices[driver_name]
            driver_mod = plc.modules_w_pins[driver_plc_idx]
            weight = driver_mod.get_weight()

            pins_in_net = []
            all_pin_names = [driver_name] + sinks
            macros_in_net = set()

            for pin_name in all_pin_names:
                parent = pin_name.split("/")[0]
                if pin_name in pin_offsets:
                    bidx, xo, yo = pin_offsets[pin_name]
                    pins_in_net.append((bidx, xo, yo))
                    macros_in_net.add(bidx)
                elif parent in self._plc_to_hard:
                    bidx = self._plc_to_hard[parent]
                    pins_in_net.append((bidx, 0.0, 0.0))
                    macros_in_net.add(bidx)
                elif parent in self._plc_to_port:
                    px, py = port_positions[parent]
                    pins_in_net.append((-1, px, py))
                elif parent in self._plc_to_soft:
                    si = self._plc_to_soft[parent]
                    bidx = n_hard + si
                    pins_in_net.append((bidx, 0.0, 0.0))
                    macros_in_net.add(bidx)

            if len(pins_in_net) < 2:
                continue

            # Record driver pin index in flat array
            driver_flat = len(flat_pins)
            for p in pins_in_net:
                flat_pins.append(p)
            net_starts.append(len(flat_pins))

            net_weights.append(weight)
            net_drivers.append(driver_flat)

            for bidx in macros_in_net:
                macro_nets[bidx].append(net_id)

            net_id += 1

        self.n_nets = net_id
        self.net_weight = np.array(net_weights, dtype=np.float64)
        self.net_starts = np.array(net_starts, dtype=np.int32)
        self.net_driver = np.array(net_drivers, dtype=np.int32)
        self.macro_nets = macro_nets

        # Reverse map: net -> list of macro bidxs in that net
        net_macros = [[] for _ in range(net_id)]
        for bidx in range(n_total):
            for nid in macro_nets[bidx]:
                net_macros[nid].append(bidx)
        self.net_macros = net_macros

        # Flat pin arrays
        self.pin_macro = np.array([p[0] for p in flat_pins], dtype=np.int32)
        self.pin_xoff = np.array([p[1] for p in flat_pins], dtype=np.float64)
        self.pin_yoff = np.array([p[2] for p in flat_pins], dtype=np.float64)
        self.pin_is_macro = self.pin_macro >= 0
        self.n_pins = len(flat_pins)

        # Current pin positions
        # float32 to match PlacementCost pin position precision
        self.pin_x = np.zeros(self.n_pins, dtype=np.float32)
        self.pin_y = np.zeros(self.n_pins, dtype=np.float32)
        self._recompute_pin_positions()

        # --- Compute initial costs ---
        self.net_hpwl = np.zeros(self.n_nets, dtype=np.float64)
        self.total_hpwl = 0.0
        self.wirelength_cost = 0.0

        self.grid_density = np.zeros(self.n_cells, dtype=np.float64)
        self.density_cost = 0.0
        self.density_cnt = max(1, math.floor(self.n_cells * 0.1))

        # Congestion state (Phase 2-3, initialized to 0 for now)
        self.V_routing_raw = np.zeros(self.n_cells, dtype=np.float64)
        self.H_routing_raw = np.zeros(self.n_cells, dtype=np.float64)
        self.V_macro_raw = np.zeros(self.n_cells, dtype=np.float64)
        self.H_macro_raw = np.zeros(self.n_cells, dtype=np.float64)
        self.congestion_cost = 0.0
        self._congestion_ready = False  # set True after Phase 2-3 implementation

        # Per-net routing cache and macro blockage cache
        self.net_routing_cache = [[] for _ in range(self.n_nets)]
        self.macro_blockage_cache = {}
        self._current_entries = []
        self.V_routing_smooth = np.zeros(self.n_cells, dtype=np.float64)
        self.H_routing_smooth = np.zeros(self.n_cells, dtype=np.float64)

        self._full_recompute_wl()
        self._full_recompute_density()
        self._full_recompute_congestion()

        # Undo state
        self._undo = None

    def _recompute_pin_positions(self):
        """Recompute all pin positions from macro positions + offsets."""
        for i in range(self.n_pins):
            m = self.pin_macro[i]
            if m >= 0:
                self.pin_x[i] = self.macro_pos[m, 0] + self.pin_xoff[i]
                self.pin_y[i] = self.macro_pos[m, 1] + self.pin_yoff[i]
            else:
                self.pin_x[i] = self.pin_xoff[i]
                self.pin_y[i] = self.pin_yoff[i]

    def _update_pins_for_macro(self, macro_idx):
        """Update pin positions for all pins belonging to a specific macro."""
        mx, my = self.macro_pos[macro_idx]
        for i in range(self.n_pins):
            if self.pin_macro[i] == macro_idx:
                self.pin_x[i] = mx + self.pin_xoff[i]
                self.pin_y[i] = my + self.pin_yoff[i]

    # --- Wirelength ---

    def _full_recompute_wl(self):
        """Full wirelength computation from scratch."""
        self.total_hpwl = 0.0
        for net_id in range(self.n_nets):
            s = self.net_starts[net_id]
            e = self.net_starts[net_id + 1]
            xs = self.pin_x[s:e]
            ys = self.pin_y[s:e]
            hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            self.net_hpwl[net_id] = hpwl
            self.total_hpwl += self.net_weight[net_id] * hpwl
        self.wirelength_cost = self.total_hpwl / ((self.cw + self.ch) * self.net_cnt)

    def _update_wl_for_nets(self, net_ids):
        """Incrementally update wirelength for affected nets."""
        for net_id in net_ids:
            old_contrib = self.net_weight[net_id] * self.net_hpwl[net_id]
            s = self.net_starts[net_id]
            e = self.net_starts[net_id + 1]
            xs = self.pin_x[s:e]
            ys = self.pin_y[s:e]
            hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            self.net_hpwl[net_id] = hpwl
            new_contrib = self.net_weight[net_id] * hpwl
            self.total_hpwl += new_contrib - old_contrib
        self.wirelength_cost = self.total_hpwl / ((self.cw + self.ch) * self.net_cnt)

    # --- Density ---

    def __get_grid_cell_location(self, x, y):
        """Replicate patched PlacementCost.__get_grid_cell_location with clamping.
        Preserves input type (float32 vs float64) for division to match PlacementCost behavior."""
        row = math.floor(y / self.grid_height)
        col = math.floor(x / self.grid_width)
        row = max(0, min(row, self.grid_row - 1))
        col = max(0, min(col, self.grid_col - 1))
        return row, col

    def _macro_density_cells(self, cx, cy, w, h):
        """Compute list of (flat_cell_idx, overlap_area) for a macro at (cx,cy)."""
        half_w, half_h = w / 2, h / 2
        x_min, y_min = cx - half_w, cy - half_h
        x_max, y_max = cx + half_w, cy + half_h

        bl_row, bl_col = self.__get_grid_cell_location(x_min, y_min)
        ur_row, ur_col = self.__get_grid_cell_location(x_max, y_max)

        result = []
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
                    result.append((r * self.grid_col + c, area))
        return result

    def _full_recompute_density(self):
        """Full density computation from scratch."""
        self.grid_density[:] = 0.0
        n_total = len(self.macro_w)
        for i in range(n_total):
            cx, cy = self.macro_pos[i]
            w, h = self.macro_w[i], self.macro_h[i]
            if w <= 0 or h <= 0:
                continue
            for cell_idx, area in self._macro_density_cells(cx, cy, w, h):
                self.grid_density[cell_idx] += area
        # Normalize by grid area
        self.grid_density /= self.grid_area
        self._recompute_density_cost()

    def _recompute_density_cost(self):
        """Recompute density cost from current grid_density array."""
        # Replicate PlacementCost.get_density_cost() exactly:
        # Sort nonzero cells descending, take top density_cnt, avg, multiply by 0.5
        if self.n_cells < 10:
            nonzero = self.grid_density[self.grid_density != 0.0]
            if len(nonzero) > 0:
                self.density_cost = 0.5 * float(nonzero.sum() / len(nonzero))
            else:
                self.density_cost = 0.0
            return

        occupied = np.sort(self.grid_density[self.grid_density != 0.0])[::-1]
        cnt = self.density_cnt
        actual = min(cnt, len(occupied))
        if actual == 0:
            self.density_cost = 0.0
        else:
            self.density_cost = 0.5 * float(occupied[:actual].sum() / cnt)

    def _update_density_for_macro(self, macro_idx, old_x, old_y, new_x, new_y):
        """Incrementally update density when one macro moves."""
        w, h = self.macro_w[macro_idx], self.macro_h[macro_idx]
        if w <= 0 or h <= 0:
            return

        # Remove old contribution
        for cell_idx, area in self._macro_density_cells(old_x, old_y, w, h):
            self.grid_density[cell_idx] -= area / self.grid_area

        # Add new contribution
        for cell_idx, area in self._macro_density_cells(new_x, new_y, w, h):
            self.grid_density[cell_idx] += area / self.grid_area

        self._recompute_density_cost()

    # --- Congestion routing (ported verbatim from plc_client_os.py) ---
    # Each routing function writes to V/H_routing_raw AND appends to
    # self._current_entries for the per-net routing cache.

    def __overlap_dist(self, block_i, block_j):
        """plc_client_os.py L981 — block = (x_min, y_min, x_max, y_max)"""
        x_diff = min(block_i[2], block_j[2]) - max(block_i[0], block_j[0])
        y_diff = min(block_i[3], block_j[3]) - max(block_i[1], block_j[1])
        if x_diff > 0 and y_diff > 0:
            return x_diff, y_diff
        return 0, 0

    def __two_pin_net_routing(self, source_gcell, node_gcells, weight):
        """plc_client_os.py L1269 — verbatim"""
        temp_gcell = list(node_gcells)
        if temp_gcell[0] == source_gcell:
            sink_gcell = temp_gcell[1]
        else:
            sink_gcell = temp_gcell[0]

        row_min = min(sink_gcell[0], source_gcell[0])
        row_max = max(sink_gcell[0], source_gcell[0])
        col_min = min(sink_gcell[1], source_gcell[1])
        col_max = max(sink_gcell[1], source_gcell[1])

        # H routing
        for col_idx in range(col_min, col_max, 1):
            col = col_idx
            row = source_gcell[0]
            idx = row * self.grid_col + col
            self.H_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, True))

        # V routing
        for row_idx in range(row_min, row_max, 1):
            row = row_idx
            col = sink_gcell[1]
            idx = row * self.grid_col + col
            self.V_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, False))

    def __l_routing(self, node_gcells, weight):
        """plc_client_os.py L1299 — verbatim"""
        node_gcells.sort(key=lambda x: (x[1], x[0]))
        y1, x1 = node_gcells[0]
        y2, x2 = node_gcells[1]
        y3, x3 = node_gcells[2]
        # H routing (x1, y1) to (x2, y1)
        for col in range(x1, x2):
            row = y1
            idx = row * self.grid_col + col
            self.H_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, True))

        # H routing (x2, y2) to (x2, y3)
        for col in range(x2, x3):
            row = y2
            idx = row * self.grid_col + col
            self.H_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, True))

        # V routing (x2, min(y1, y2)) to (x2, max(y1, y2))
        for row in range(min(y1, y2), max(y1, y2)):
            col = x2
            idx = row * self.grid_col + col
            self.V_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, False))

        # V routing (x3, min(y2, y3)) to (x3, max(y2, y3))
        for row in range(min(y2, y3), max(y2, y3)):
            col = x3
            idx = row * self.grid_col + col
            self.V_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, False))

    def __t_routing(self, node_gcells, weight):
        """plc_client_os.py L1328 — verbatim"""
        node_gcells.sort()
        y1, x1 = node_gcells[0]
        y2, x2 = node_gcells[1]
        y3, x3 = node_gcells[2]
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)

        # H routing (xmin, y2) to (xmax, y2)
        for col in range(xmin, xmax):
            row = y2
            idx = row * self.grid_col + col
            self.H_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, True))

        # V routing (x1, y1) to (x1, y2)
        for row in range(min(y1, y2), max(y1, y2)):
            col = x1
            idx = row * self.grid_col + col
            self.V_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, False))

        # V routing (x3, y3) to (x3, y2)
        for row in range(min(y2, y3), max(y2, y3)):
            col = x3
            idx = row * self.grid_col + col
            self.V_routing_raw[idx] += weight
            self._current_entries.append((idx, weight, False))

    def __three_pin_net_routing(self, node_gcells, weight):
        """plc_client_os.py L1354 — verbatim"""
        temp_gcell = list(node_gcells)
        temp_gcell.sort(key=lambda x: (x[1], x[0]))
        y1, x1 = temp_gcell[0]
        y2, x2 = temp_gcell[1]
        y3, x3 = temp_gcell[2]

        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self.__l_routing(temp_gcell, weight)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for col_idx in range(x1, x2, 1):
                row = y1
                col = col_idx
                idx = row * self.grid_col + col
                self.H_routing_raw[idx] += weight
                self._current_entries.append((idx, weight, True))

            for row_idx in range(y1, max(y2, y3)):
                col = x2
                row = row_idx
                idx = row * self.grid_col + col
                self.V_routing_raw[idx] += weight
                self._current_entries.append((idx, weight, False))
        elif y2 == y3:
            for col in range(x1, x2):
                row = y1
                idx = row * self.grid_col + col
                self.H_routing_raw[idx] += weight
                self._current_entries.append((idx, weight, True))

            for col in range(x2, x3):
                row = y2
                idx = row * self.grid_col + col
                self.H_routing_raw[idx] += weight
                self._current_entries.append((idx, weight, True))

            for row in range(min(y2, y1), max(y2, y1)):
                col = x2
                idx = row * self.grid_col + col
                self.V_routing_raw[idx] += weight
                self._current_entries.append((idx, weight, False))
        else:
            self.__t_routing(temp_gcell, weight)

    def __split_net(self, source_gcell, node_gcells):
        """plc_client_os.py L1486 — verbatim"""
        splitted_netlist = []
        for node_gcell in node_gcells:
            if node_gcell != source_gcell:
                splitted_netlist.append({source_gcell, node_gcell})
        return splitted_netlist

    def __macro_route_over_grid_cell(self, mod_x, mod_y, mod_w, mod_h):
        """plc_client_os.py L1392 — verbatim, uses tuples instead of Block objects.
        Returns list of (flat_idx, v_amount, h_amount) for cache tracking.
        Uses float32 arithmetic to match PlacementCost's type promotion behavior."""
        # Match plc behavior: positions are np.float32, sizes are Python float.
        # np.float32 + Python_float → np.float32 (numpy promotion rules).
        # CRITICAL: do NOT convert to float64 — grid cell boundary assignment depends
        # on float32 arithmetic giving exact results at boundaries.
        mod_x = np.float32(mod_x)
        mod_y = np.float32(mod_y)
        ur = (mod_x + (mod_w / 2), mod_y + (mod_h / 2))
        bl = (mod_x - (mod_w / 2), mod_y - (mod_h / 2))

        module_block = (bl[0], bl[1], ur[0], ur[1])  # x_min, y_min, x_max, y_max

        ur_row, ur_col = self.__get_grid_cell_location(*ur)
        bl_row, bl_col = self.__get_grid_cell_location(*bl)

        if ur_row >= 0 and ur_col >= 0:
            if bl_row < 0:
                bl_row = 0
            if bl_col < 0:
                bl_col = 0
        else:
            return []

        if bl_row >= 0 and bl_col >= 0:
            if ur_row > self.grid_row - 1:
                ur_row = self.grid_row - 1
            if ur_col > self.grid_col - 1:
                ur_col = self.grid_col - 1
        else:
            return []

        if_PARTIAL_OVERLAP_VERTICAL = False
        if_PARTIAL_OVERLAP_HORIZONTAL = False
        entries = []

        for r_i in range(bl_row, ur_row + 1):
            for c_i in range(bl_col, ur_col + 1):
                grid_cell_block = (c_i * self.grid_width, r_i * self.grid_height,
                                   (c_i + 1) * self.grid_width, (r_i + 1) * self.grid_height)

                x_dist, y_dist = self.__overlap_dist(module_block, grid_cell_block)

                if ur_row != bl_row:
                    if (r_i == bl_row and abs(y_dist - self.grid_height) > 1e-5) or \
                       (r_i == ur_row and abs(y_dist - self.grid_height) > 1e-5):
                        if_PARTIAL_OVERLAP_VERTICAL = True

                if ur_col != bl_col:
                    if (c_i == bl_col and abs(x_dist - self.grid_width) > 1e-5) or \
                       (c_i == ur_col and abs(x_dist - self.grid_width) > 1e-5):
                        if_PARTIAL_OVERLAP_HORIZONTAL = True

                flat = r_i * self.grid_col + c_i
                v_amt = x_dist * self.vrouting_alloc
                h_amt = y_dist * self.hrouting_alloc
                self.V_macro_raw[flat] += v_amt
                self.H_macro_raw[flat] += h_amt
                entries.append((flat, v_amt, h_amt))

        if if_PARTIAL_OVERLAP_VERTICAL:
            for c_i in range(bl_col, ur_col + 1):
                r_i = ur_row
                grid_cell_block = (c_i * self.grid_width, r_i * self.grid_height,
                                   (c_i + 1) * self.grid_width, (r_i + 1) * self.grid_height)
                x_dist, y_dist = self.__overlap_dist(module_block, grid_cell_block)
                flat = r_i * self.grid_col + c_i
                v_sub = x_dist * self.vrouting_alloc
                self.V_macro_raw[flat] -= v_sub
                entries.append((flat, -v_sub, 0.0))

        if if_PARTIAL_OVERLAP_HORIZONTAL:
            for r_i in range(bl_row, ur_row + 1):
                c_i = ur_col
                grid_cell_block = (c_i * self.grid_width, r_i * self.grid_height,
                                   (c_i + 1) * self.grid_width, (r_i + 1) * self.grid_height)
                x_dist, y_dist = self.__overlap_dist(module_block, grid_cell_block)
                flat = r_i * self.grid_col + c_i
                h_sub = y_dist * self.hrouting_alloc
                self.H_macro_raw[flat] -= h_sub
                entries.append((flat, 0.0, -h_sub))

        return entries

    def __smooth_routing_cong(self):
        """plc_client_os.py L1608 — vectorized smoothing of V/H routing."""
        gc, gr, sr = self.grid_col, self.grid_row, self.smooth_range
        V_norm = (self.V_routing_raw / self.grid_v_routes).reshape(gr, gc)
        H_norm = (self.H_routing_raw / self.grid_h_routes).reshape(gr, gc)

        # Smooth V horizontally: each cell distributes value/count to [col-sr, col+sr]
        temp_V = np.zeros((gr, gc), dtype=np.float64)
        for col in range(gc):
            lp = max(0, col - sr)
            rp = min(gc - 1, col + sr)
            cnt = rp - lp + 1
            temp_V[:, lp:rp + 1] += V_norm[:, col:col + 1] / cnt

        # Smooth H vertically: each cell distributes value/count to [row-sr, row+sr]
        temp_H = np.zeros((gr, gc), dtype=np.float64)
        for row in range(gr):
            lp = max(0, row - sr)
            up = min(gr - 1, row + sr)
            cnt = up - lp + 1
            temp_H[lp:up + 1, :] += H_norm[row:row + 1, :] / cnt

        self.V_routing_smooth = temp_V.ravel()
        self.H_routing_smooth = temp_H.ravel()

    def _recompute_congestion_cost(self):
        """Recompute congestion from raw arrays: normalize, smooth, combine, ABU."""
        self.__smooth_routing_cong()

        V_macro_norm = self.V_macro_raw / self.grid_v_routes
        H_macro_norm = self.H_macro_raw / self.grid_h_routes

        V_total = self.V_routing_smooth + V_macro_norm
        H_total = self.H_routing_smooth + H_macro_norm

        # ABU: top 5% of combined V+H
        combined = np.concatenate([V_total, H_total])
        sorted_vals = np.sort(combined)[::-1]
        cnt = max(1, math.floor(len(combined) * 0.05))
        self.congestion_cost = float(sorted_vals[:cnt].sum() / cnt)

    def _route_net(self, net_id):
        """Route a single net, recording entries in cache."""
        self._current_entries = []

        s = self.net_starts[net_id]
        e = self.net_starts[net_id + 1]
        driver_pin = self.net_driver[net_id]
        weight = self.net_weight[net_id]

        # Routing weight: ports use 1, macro pins use pin weight only if > 1
        # (matching get_routing() behavior: weight defaults to 1, only overridden for MACRO_PIN with weight > 1)
        routing_weight = weight if weight > 1 else 1.0

        # Compute grid cell locations as a SET (dedup coincident pins)
        node_gcells = set()
        source_gcell = self.__get_grid_cell_location(self.pin_x[driver_pin], self.pin_y[driver_pin])
        node_gcells.add(source_gcell)

        for i in range(s, e):
            gcell = self.__get_grid_cell_location(self.pin_x[i], self.pin_y[i])
            node_gcells.add(gcell)

        if len(node_gcells) == 2:
            self.__two_pin_net_routing(source_gcell=source_gcell, node_gcells=node_gcells, weight=routing_weight)
        elif len(node_gcells) == 3:
            self.__three_pin_net_routing(node_gcells=node_gcells, weight=routing_weight)
        elif len(node_gcells) > 3:
            for curr_net in self.__split_net(source_gcell=source_gcell, node_gcells=node_gcells):
                self.__two_pin_net_routing(source_gcell=source_gcell, node_gcells=curr_net, weight=routing_weight)

        self.net_routing_cache[net_id] = list(self._current_entries)

    def _unroute_net(self, net_id):
        """Remove a net's routing contribution from V/H_routing_raw."""
        for flat_idx, amount, is_H in self.net_routing_cache[net_id]:
            if is_H:
                self.H_routing_raw[flat_idx] -= amount
            else:
                self.V_routing_raw[flat_idx] -= amount
        self.net_routing_cache[net_id] = []

    def _full_recompute_congestion(self):
        """Full congestion computation from scratch."""
        self.V_routing_raw[:] = 0.0
        self.H_routing_raw[:] = 0.0
        self.V_macro_raw[:] = 0.0
        self.H_macro_raw[:] = 0.0

        # Route all nets
        for net_id in range(self.n_nets):
            self._route_net(net_id)

        # Macro blockage for all hard macros
        self.macro_blockage_cache = {}
        for i in range(self.n_hard):
            cx, cy = self.macro_pos[i]
            # Convert to Python float to match PlacementCost type promotion:
            # np.float32(pos) + Python_float(size/2) → np.float32
            w, h = float(self.macro_w[i]), float(self.macro_h[i])
            if w > 0 and h > 0:
                entries = self.__macro_route_over_grid_cell(cx, cy, w, h)
                self.macro_blockage_cache[i] = entries

        self._recompute_congestion_cost()
        self._congestion_ready = True

    # --- Proxy cost ---

    def get_proxy_cost(self):
        """Return current proxy cost."""
        return 1.0 * self.wirelength_cost + 0.5 * self.density_cost + 0.5 * self.congestion_cost

    def sync_positions(self, hard_pos):
        """Set hard macro positions and recompute all costs."""
        self.macro_pos[:self.n_hard] = np.asarray(hard_pos, dtype=np.float32)
        self._recompute_pin_positions()
        self._full_recompute_wl()
        self._full_recompute_density()
        self._full_recompute_congestion()

    def move_macro(self, macro_idx, new_x, new_y):
        """Move a hard macro, incrementally update costs, return new proxy cost."""
        old_x, old_y = self.macro_pos[macro_idx, 0], self.macro_pos[macro_idx, 1]

        # Save undo state
        affected_nets = self.macro_nets[macro_idx]
        old_net_hpwl = {nid: self.net_hpwl[nid] for nid in affected_nets}
        old_total_hpwl = self.total_hpwl
        old_wl_cost = self.wirelength_cost

        # Save affected pin positions
        affected_pins = [i for i in range(self.n_pins) if self.pin_macro[i] == macro_idx]
        old_pin_x = {i: self.pin_x[i] for i in affected_pins}
        old_pin_y = {i: self.pin_y[i] for i in affected_pins}

        # Save density state (affected cells)
        w, h = self.macro_w[macro_idx], self.macro_h[macro_idx]
        old_density_cells = {}
        if w > 0 and h > 0:
            for cell_idx, _ in self._macro_density_cells(old_x, old_y, w, h):
                old_density_cells[cell_idx] = self.grid_density[cell_idx]
            for cell_idx, _ in self._macro_density_cells(new_x, new_y, w, h):
                if cell_idx not in old_density_cells:
                    old_density_cells[cell_idx] = self.grid_density[cell_idx]
        old_density_cost = self.density_cost

        # Save congestion state (full arrays — 57KB, fast to copy)
        old_V_routing = self.V_routing_raw.copy()
        old_H_routing = self.H_routing_raw.copy()
        old_V_macro = self.V_macro_raw.copy()
        old_H_macro = self.H_macro_raw.copy()
        old_V_smooth = self.V_routing_smooth.copy()
        old_H_smooth = self.H_routing_smooth.copy()
        old_cong_cost = self.congestion_cost
        old_net_routing_cache = {nid: list(self.net_routing_cache[nid]) for nid in affected_nets}
        old_macro_blockage = list(self.macro_blockage_cache.get(macro_idx, []))

        self._undo = {
            'macro_idx': macro_idx,
            'old_x': old_x, 'old_y': old_y,
            'old_net_hpwl': old_net_hpwl,
            'old_total_hpwl': old_total_hpwl,
            'old_wl_cost': old_wl_cost,
            'old_pin_x': old_pin_x,
            'old_pin_y': old_pin_y,
            'old_density_cells': old_density_cells,
            'old_density_cost': old_density_cost,
            'old_V_routing': old_V_routing,
            'old_H_routing': old_H_routing,
            'old_V_macro': old_V_macro,
            'old_H_macro': old_H_macro,
            'old_V_smooth': old_V_smooth,
            'old_H_smooth': old_H_smooth,
            'old_cong_cost': old_cong_cost,
            'old_net_routing_cache': old_net_routing_cache,
            'old_macro_blockage': old_macro_blockage,
        }

        # Apply move (float32 to match plc precision)
        self.macro_pos[macro_idx, 0] = np.float32(new_x)
        self.macro_pos[macro_idx, 1] = np.float32(new_y)
        self._update_pins_for_macro(macro_idx)

        # Incremental wirelength + density
        self._update_wl_for_nets(affected_nets)
        self._update_density_for_macro(macro_idx, old_x, old_y, new_x, new_y)

        # Incremental congestion
        # 1. Remove old net routing for affected nets
        for nid in affected_nets:
            self._unroute_net(nid)

        # 2. Remove old macro blockage (hard macros only)
        if macro_idx < self.n_hard:
            for flat, v_amt, h_amt in self.macro_blockage_cache.get(macro_idx, []):
                self.V_macro_raw[flat] -= v_amt
                self.H_macro_raw[flat] -= h_amt

        # 3. Re-route affected nets with new pin positions
        for nid in affected_nets:
            self._route_net(nid)

        # 4. Add new macro blockage
        if macro_idx < self.n_hard:
            w_f, h_f = float(self.macro_w[macro_idx]), float(self.macro_h[macro_idx])
            if w_f > 0 and h_f > 0:
                entries = self.__macro_route_over_grid_cell(
                    self.macro_pos[macro_idx, 0], self.macro_pos[macro_idx, 1], w_f, h_f)
                self.macro_blockage_cache[macro_idx] = entries

        # 5. Re-smooth and recompute congestion cost
        self._recompute_congestion_cost()

        return self.get_proxy_cost()

    def undo_move(self):
        """Restore state before last move_macro call."""
        if self._undo is None:
            return

        u = self._undo
        macro_idx = u['macro_idx']

        # Restore macro position
        self.macro_pos[macro_idx, 0] = u['old_x']
        self.macro_pos[macro_idx, 1] = u['old_y']

        # Restore pin positions
        for i, x in u['old_pin_x'].items():
            self.pin_x[i] = x
        for i, y in u['old_pin_y'].items():
            self.pin_y[i] = y

        # Restore wirelength
        for nid, hpwl in u['old_net_hpwl'].items():
            self.net_hpwl[nid] = hpwl
        self.total_hpwl = u['old_total_hpwl']
        self.wirelength_cost = u['old_wl_cost']

        # Restore density
        for cell_idx, val in u['old_density_cells'].items():
            self.grid_density[cell_idx] = val
        self.density_cost = u['old_density_cost']

        # Restore congestion (swap arrays for O(1))
        self.V_routing_raw[:] = u['old_V_routing']
        self.H_routing_raw[:] = u['old_H_routing']
        self.V_macro_raw[:] = u['old_V_macro']
        self.H_macro_raw[:] = u['old_H_macro']
        self.V_routing_smooth[:] = u['old_V_smooth']
        self.H_routing_smooth[:] = u['old_H_smooth']
        self.congestion_cost = u['old_cong_cost']
        for nid, cache in u['old_net_routing_cache'].items():
            self.net_routing_cache[nid] = cache
        self.macro_blockage_cache[macro_idx] = u['old_macro_blockage']

        self._undo = None


# ---------------------------------------------------------------------------
# Quadratic global placement (Laplacian solve)
# ---------------------------------------------------------------------------

def _quadratic_place(benchmark, plc):
    """Quadratic global placement via Laplacian solve.

    Builds spring connections from plc.nets (clique model). For each net:
      - Hard macros connected to other hard macros: spring forces (movable-movable)
      - Hard macros connected to fixed pins (ports/soft macros): right-hand-side forces

    Solves L x = bx and L y = by where L is the Laplacian of the spring graph
    over movable hard macros only.

    Uses plc.nets (not benchmark.net_nodes which is empty in .pt files for IBM).
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
    soft_pos = benchmark.macro_positions[n_hard:].numpy().astype(np.float64)

    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return init_pos

    # Map hard macro index → index in movable subset
    movable_map = {int(idx): k for k, idx in enumerate(movable_idx)}
    n_mov = len(movable_idx)

    # Build name → benchmark idx maps for plc traversal
    plc_to_hard = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        plc_to_hard[plc.modules_w_pins[plc_idx].get_name()] = bidx

    plc_to_soft = {}
    for bidx, plc_idx in enumerate(plc.soft_macro_indices):
        plc_to_soft[plc.modules_w_pins[plc_idx].get_name()] = bidx

    port_pos = {}
    for plc_idx in plc.port_indices:
        mod = plc.modules_w_pins[plc_idx]
        port_pos[mod.get_name()] = mod.get_pos()

    # Pin offsets for hard macros (used to attribute pins to parent macros)
    pin_offsets = {}
    for _, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == 'MACRO_PIN':
            parent = mod.get_name().split("/")[0]
            if parent in plc_to_hard:
                pin_offsets[mod.get_name()] = plc_to_hard[parent]

    # Laplacian and RHS — dense numpy arrays (n_mov is small, ≤ ~800)
    L = np.zeros((n_mov, n_mov), dtype=np.float64)
    bx = np.zeros(n_mov, dtype=np.float64)
    by = np.zeros(n_mov, dtype=np.float64)
    weight_sum = 0.0
    n_edges = 0

    for driver_name, sinks in plc.nets.items():
        # Collect movable hard macros and fixed (x,y) anchors in this net
        mov_in_net = set()
        fixed_anchors = []  # list of (fx, fy)

        all_pins = [driver_name] + sinks
        for pin_name in all_pins:
            parent = pin_name.split("/")[0]
            if pin_name in pin_offsets:
                bidx = pin_offsets[pin_name]
                if movable[bidx]:
                    mov_in_net.add(bidx)
                else:
                    cx, cy = init_pos[bidx, 0], init_pos[bidx, 1]
                    fixed_anchors.append((cx, cy))
            elif parent in plc_to_hard:
                bidx = plc_to_hard[parent]
                if movable[bidx]:
                    mov_in_net.add(bidx)
                else:
                    cx, cy = init_pos[bidx, 0], init_pos[bidx, 1]
                    fixed_anchors.append((cx, cy))
            elif parent in port_pos:
                px, py = port_pos[parent]
                fixed_anchors.append((float(px), float(py)))
            elif parent in plc_to_soft:
                si = plc_to_soft[parent]
                fixed_anchors.append((float(soft_pos[si, 0]), float(soft_pos[si, 1])))

        n_pins = len(mov_in_net) + len(fixed_anchors)
        if n_pins < 2 or len(mov_in_net) == 0:
            continue

        # Net weight (clique model: w / (k - 1))
        w = 1.0 / max(n_pins - 1, 1)
        mov_list = sorted(mov_in_net)

        # Movable–movable spring connections
        for ai in range(len(mov_list)):
            for bi in range(ai + 1, len(mov_list)):
                a = movable_map[mov_list[ai]]
                b = movable_map[mov_list[bi]]
                L[a, b] -= w
                L[b, a] -= w
                L[a, a] += w
                L[b, b] += w
                weight_sum += w
                n_edges += 1

        # Movable–fixed pin forces (RHS)
        for fx, fy in fixed_anchors:
            for m in mov_list:
                k = movable_map[m]
                L[k, k] += w
                bx[k] += w * fx
                by[k] += w * fy
                weight_sum += w

    # Anchor: pull all movable macros toward canvas center (handles isolated nodes)
    if weight_sum < 1e-12:
        anchor = 1.0
    else:
        anchor = 0.01 * (weight_sum / max(n_mov, 1))
    cx_center, cy_center = cw / 2, ch / 2
    for k in range(n_mov):
        L[k, k] += anchor
        bx[k] += anchor * cx_center
        by[k] += anchor * cy_center

    # Solve
    try:
        x_sol = np.linalg.solve(L, bx)
        y_sol = np.linalg.solve(L, by)
    except np.linalg.LinAlgError:
        return init_pos

    pos = init_pos.copy()
    for k, idx in enumerate(movable_idx):
        pos[idx, 0] = np.clip(x_sol[k], half_w[idx], cw - half_w[idx])
        pos[idx, 1] = np.clip(y_sol[k], half_h[idx], ch - half_h[idx])

    return pos


# ---------------------------------------------------------------------------
# Weighted centroid placement (Jacobi iteration on net neighbors)
# ---------------------------------------------------------------------------

def _centroid_place(benchmark, plc, n_iters=5):
    """Iterative weighted-centroid placement.

    For each movable hard macro, repeatedly set its position to the weighted
    average of all positions it's connected to via nets. Fixed pins (ports,
    soft macros, fixed hard macros) anchor the system.

    This is Jacobi iteration on the same system the quadratic placer solves
    in closed form, but the iterative form is simpler and converges enough in
    a few passes for use as a legalization seed.
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
    soft_pos = benchmark.macro_positions[n_hard:].numpy().astype(np.float64)

    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return init_pos

    # Build name → benchmark idx maps
    plc_to_hard = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        plc_to_hard[plc.modules_w_pins[plc_idx].get_name()] = bidx

    plc_to_soft = {}
    for bidx, plc_idx in enumerate(plc.soft_macro_indices):
        plc_to_soft[plc.modules_w_pins[plc_idx].get_name()] = bidx

    port_pos = {}
    for plc_idx in plc.port_indices:
        mod = plc.modules_w_pins[plc_idx]
        port_pos[mod.get_name()] = mod.get_pos()

    pin_offsets = {}
    for _, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == 'MACRO_PIN':
            parent = mod.get_name().split("/")[0]
            if parent in plc_to_hard:
                pin_offsets[mod.get_name()] = plc_to_hard[parent]

    # Pre-compute per-macro neighbor lists.
    # For each movable macro: list of (neighbor_type, neighbor_data, weight)
    #   neighbor_type='mov': data is hard macro idx (looked up in pos each iter)
    #   neighbor_type='fix': data is (fx, fy) tuple (precomputed)
    macro_neighbors = [[] for _ in range(n_hard)]

    for driver_name, sinks in plc.nets.items():
        mov_in_net = set()
        fixed_anchors = []  # list of (fx, fy)
        fixed_movable = set()  # fixed hard macros (not movable but in our index range)

        all_pins = [driver_name] + sinks
        for pin_name in all_pins:
            parent = pin_name.split("/")[0]
            if pin_name in pin_offsets:
                bidx = pin_offsets[pin_name]
                if movable[bidx]:
                    mov_in_net.add(bidx)
                else:
                    fixed_anchors.append((float(init_pos[bidx, 0]), float(init_pos[bidx, 1])))
            elif parent in plc_to_hard:
                bidx = plc_to_hard[parent]
                if movable[bidx]:
                    mov_in_net.add(bidx)
                else:
                    fixed_anchors.append((float(init_pos[bidx, 0]), float(init_pos[bidx, 1])))
            elif parent in port_pos:
                px, py = port_pos[parent]
                fixed_anchors.append((float(px), float(py)))
            elif parent in plc_to_soft:
                si = plc_to_soft[parent]
                fixed_anchors.append((float(soft_pos[si, 0]), float(soft_pos[si, 1])))

        n_pins = len(mov_in_net) + len(fixed_anchors)
        if n_pins < 2 or len(mov_in_net) == 0:
            continue

        # Net weight (clique model)
        w = 1.0 / max(n_pins - 1, 1)
        mov_list = sorted(mov_in_net)

        # Each movable macro in this net is connected to all OTHER pins in the net
        for m in mov_list:
            for other in mov_list:
                if other != m:
                    macro_neighbors[m].append(('mov', other, w))
            for fx, fy in fixed_anchors:
                macro_neighbors[m].append(('fix', (fx, fy), w))

    # Iterative weighted centroid update
    pos = init_pos.copy()
    for it in range(n_iters):
        new_pos = pos.copy()
        for idx in movable_idx:
            neighbors = macro_neighbors[idx]
            if not neighbors:
                continue
            sum_x = 0.0
            sum_y = 0.0
            sum_w = 0.0
            for ntype, ndata, w in neighbors:
                if ntype == 'mov':
                    nx = pos[ndata, 0]
                    ny = pos[ndata, 1]
                else:
                    nx, ny = ndata
                sum_x += w * nx
                sum_y += w * ny
                sum_w += w
            if sum_w > 1e-12:
                new_pos[idx, 0] = sum_x / sum_w
                new_pos[idx, 1] = sum_y / sum_w
        pos = new_pos

    # Clip to canvas bounds
    for idx in movable_idx:
        pos[idx, 0] = np.clip(pos[idx, 0], half_w[idx], cw - half_w[idx])
        pos[idx, 1] = np.clip(pos[idx, 1], half_h[idx], ch - half_h[idx])

    return pos


# ---------------------------------------------------------------------------
# Gradient-based global placement (finite-difference on real proxy cost)
# ---------------------------------------------------------------------------

def _gradient_place(benchmark):
    """Global placement via finite-difference gradient descent on the real
    proxy cost (including congestion).

    Each iteration:
      1. Probe each movable macro with ±eps in x and y, computing finite-diff
         gradient via incr_eval.move_macro / undo_move
      2. Apply simultaneous update to all macros via sync_positions

    Overlaps are allowed in the output — legalization handles them.
    Uses its own IncrementalEvaluator (does not share with CD).
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]
    init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()

    if len(movable_idx) == 0:
        return init_pos

    plc = _load_plc(benchmark.name)
    if plc is None:
        return init_pos

    # Own evaluator — do not share with CD
    incr_eval = IncrementalEvaluator(plc, benchmark)

    pos = init_pos.copy()
    incr_eval.sync_positions(pos)

    macro_max_dim = np.maximum(sizes[:, 0], sizes[:, 1])
    canvas_max = max(cw, ch)

    n_iters = 150
    time_budget = 300.0
    t0 = time.time()

    for it in range(n_iters):
        if time.time() - t0 > time_budget:
            print(f"  GradPlace stopped at iter {it} (time budget exceeded)")
            break

        decay = max(0.05, 1.0 - it / n_iters)
        decay_lr = max(0.02, 1.0 - it / n_iters)
        lr = 2.0 * canvas_max * decay_lr

        # --- Compute gradients for all movable macros ---
        grad = np.zeros((n_hard, 2), dtype=np.float64)

        for idx in movable_idx:
            ox = float(pos[idx, 0])
            oy = float(pos[idx, 1])
            eps = 0.5 * macro_max_dim[idx] * decay
            if eps < 1e-6:
                continue

            # +x probe
            x_p = min(ox + eps, cw - half_w[idx])
            x_m = max(ox - eps, half_w[idx])
            dx_span = x_p - x_m
            if dx_span > 1e-6:
                c_xp = incr_eval.move_macro(idx, x_p, oy)
                incr_eval.undo_move()
                c_xm = incr_eval.move_macro(idx, x_m, oy)
                incr_eval.undo_move()
                grad[idx, 0] = (c_xp - c_xm) / dx_span

            # +y probe
            y_p = min(oy + eps, ch - half_h[idx])
            y_m = max(oy - eps, half_h[idx])
            dy_span = y_p - y_m
            if dy_span > 1e-6:
                c_yp = incr_eval.move_macro(idx, ox, y_p)
                incr_eval.undo_move()
                c_ym = incr_eval.move_macro(idx, ox, y_m)
                incr_eval.undo_move()
                grad[idx, 1] = (c_yp - c_ym) / dy_span

        if time.time() - t0 > time_budget:
            print(f"  GradPlace stopped during iter {it} (time budget exceeded)")
            break

        # --- Apply simultaneous update ---
        # Normalize so the largest move is at most lr
        grad_norm = np.linalg.norm(grad[movable_idx])
        if grad_norm < 1e-12:
            break

        for idx in movable_idx:
            new_x = pos[idx, 0] - lr * grad[idx, 0]
            new_y = pos[idx, 1] - lr * grad[idx, 1]
            pos[idx, 0] = np.clip(new_x, half_w[idx], cw - half_w[idx])
            pos[idx, 1] = np.clip(new_y, half_h[idx], ch - half_h[idx])

        # Sync evaluator with new positions
        incr_eval.sync_positions(pos)

        if (it + 1) % 25 == 0:
            print(f"  GradPlace iter {it + 1}: cost={incr_eval.get_proxy_cost():.6f}")

    return pos


# ---------------------------------------------------------------------------
# Analytical global placement (spectral embedding + Nesterov HPWL+density)
# ---------------------------------------------------------------------------

def _analytical_placement(benchmark, plc, max_time=60):
    """Spectral embedding + Nesterov gradient descent for a connectivity-aware seed.

    1. Build weighted adjacency from plc.nets (clique model)
    2. Laplacian eigenvectors 2,3 → initial (x,y)
    3. Nesterov on smooth HPWL (log-sum-exp) + bin density penalty
    4. Return raw positions (caller handles legalization)
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]

    if len(movable_idx) < 2:
        return benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)

    t0 = time.time()

    # --- Map plc names to benchmark indices ---
    plc_to_hard = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        plc_to_hard[plc.modules_w_pins[plc_idx].get_name()] = bidx

    plc_to_soft = {}
    for bidx, plc_idx in enumerate(plc.soft_macro_indices):
        plc_to_soft[plc.modules_w_pins[plc_idx].get_name()] = bidx

    port_pos_map = {}
    for plc_idx in plc.port_indices:
        mod = plc.modules_w_pins[plc_idx]
        port_pos_map[mod.get_name()] = mod.get_pos()

    pin_offsets = {}
    for _, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == 'MACRO_PIN':
            parent = mod.get_name().split("/")[0]
            if parent in plc_to_hard:
                xo, yo = mod.get_offset()
                pin_offsets[mod.get_name()] = (plc_to_hard[parent], xo, yo)

    # --- Build net data in flat arrays for fast gradient computation ---
    # Each pin: (macro_bidx or -1, x_offset, y_offset, is_fixed)
    flat_macro = []   # macro index (-1 if fixed)
    flat_xoff = []    # pin x offset (or absolute x for fixed)
    flat_yoff = []    # pin y offset (or absolute y for fixed)
    net_starts = [0]  # CSR start indices

    soft_pos = benchmark.macro_positions[benchmark.num_hard_macros:].numpy().astype(np.float64)

    adj_rows, adj_cols, adj_data = [], [], []  # for spectral embedding

    for driver_name, sinks in plc.nets.items():
        all_pin_names = [driver_name] + sinks
        pins = []
        hard_in_net = set()

        for pin_name in all_pin_names:
            parent = pin_name.split("/")[0]
            if pin_name in pin_offsets:
                bidx, xo, yo = pin_offsets[pin_name]
                if bidx not in hard_in_net:
                    pins.append((bidx, xo, yo))
                    hard_in_net.add(bidx)
            elif parent in plc_to_hard:
                bidx = plc_to_hard[parent]
                if bidx not in hard_in_net:
                    pins.append((bidx, 0.0, 0.0))
                    hard_in_net.add(bidx)
            elif parent in port_pos_map:
                px, py = port_pos_map[parent]
                pins.append((-1, px, py))
            elif parent in plc_to_soft:
                si = plc_to_soft[parent]
                sx, sy = float(soft_pos[si, 0]), float(soft_pos[si, 1])
                pins.append((-1, sx, sy))

        if len(pins) < 2 or len(hard_in_net) == 0:
            continue

        for bidx, xo, yo in pins:
            flat_macro.append(bidx)
            flat_xoff.append(xo)
            flat_yoff.append(yo)
        net_starts.append(len(flat_macro))

        # Adjacency for spectral embedding (clique model)
        hard_list = sorted(hard_in_net)
        k = len(hard_list)
        if k >= 2:
            w = 1.0 / (k - 1)
            for ai in range(k):
                for bi in range(ai + 1, k):
                    a, b = hard_list[ai], hard_list[bi]
                    adj_rows.extend([a, b])
                    adj_cols.extend([b, a])
                    adj_data.extend([w, w])

    n_nets = len(net_starts) - 1
    flat_macro = np.array(flat_macro, dtype=np.int32)
    flat_xoff = np.array(flat_xoff, dtype=np.float64)
    flat_yoff = np.array(flat_yoff, dtype=np.float64)
    net_starts = np.array(net_starts, dtype=np.int32)

    if n_nets == 0 or len(adj_data) == 0:
        return benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)

    # --- Step 1: Spectral embedding ---
    A = sp.coo_matrix((adj_data, (adj_rows, adj_cols)),
                       shape=(n_hard, n_hard)).tocsr()
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    L = (D - A).tocsc()

    n_eig = min(4, n_hard - 1)
    try:
        eigvals, eigvecs = eigsh(L, k=n_eig, which='SM', tol=1e-4)
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        # Skip eigenvector 0 (constant), use 1 and 2
        x_spec = eigvecs[:, min(1, n_eig - 1)]
        y_spec = eigvecs[:, min(2, n_eig - 1)]
    except Exception:
        rng = np.random.RandomState(42)
        x_spec = rng.randn(n_hard)
        y_spec = rng.randn(n_hard)

    # Scale eigenvectors to canvas
    margin_x = half_w.max() * 1.5
    margin_y = half_h.max() * 1.5

    def _norm_range(v, lo, hi):
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-10:
            return np.full_like(v, (lo + hi) / 2)
        return lo + (v - vmin) / (vmax - vmin) * (hi - lo)

    pos = np.zeros((n_hard, 2), dtype=np.float64)
    pos[:, 0] = _norm_range(x_spec, margin_x, cw - margin_x)
    pos[:, 1] = _norm_range(y_spec, margin_y, ch - margin_y)

    # Pin down immovable macros
    for i in range(n_hard):
        if not movable[i]:
            pos[i] = benchmark.macro_positions[i].numpy().astype(np.float64)

    # --- Step 2: Nesterov gradient descent on smooth HPWL + density ---
    gamma = max(0.5, min(cw, ch) * 0.02)  # LSE smoothing, scale-adaptive
    n_bins = 10
    bin_w = cw / n_bins
    bin_h = ch / n_bins
    total_macro_area = (sizes[:, 0] * sizes[:, 1]).sum()
    target_density = total_macro_area / (cw * ch)
    macro_area = sizes[:, 0] * sizes[:, 1]

    density_weight = 0.005
    lr = 1.0
    momentum = 0.9
    velocity = np.zeros_like(pos)

    for iteration in range(2000):
        if time.time() - t0 > max_time * 0.9:
            break

        # Momentum lookahead position
        y_pos = pos + momentum * velocity

        # Clip lookahead
        for i in movable_idx:
            y_pos[i, 0] = np.clip(y_pos[i, 0], half_w[i], cw - half_w[i])
            y_pos[i, 1] = np.clip(y_pos[i, 1], half_h[i], ch - half_h[i])

        grad = np.zeros_like(pos)

        # --- HPWL gradient (log-sum-exp) per net ---
        for n in range(n_nets):
            s, e = net_starts[n], net_starts[n + 1]
            n_pins = e - s
            if n_pins < 2:
                continue

            # Compute pin positions
            px = np.empty(n_pins, dtype=np.float64)
            py = np.empty(n_pins, dtype=np.float64)
            for k in range(n_pins):
                m = flat_macro[s + k]
                if m >= 0:
                    px[k] = y_pos[m, 0] + flat_xoff[s + k]
                    py[k] = y_pos[m, 1] + flat_yoff[s + k]
                else:
                    px[k] = flat_xoff[s + k]
                    py[k] = flat_yoff[s + k]

            # X gradient
            px_shift = px - px.max()
            exp_p = np.exp(px_shift / gamma)
            mx_shift = -px + px.min()
            exp_m = np.exp(mx_shift / gamma)
            sp_sum = exp_p.sum()
            sm_sum = exp_m.sum()
            gx = exp_p / sp_sum - exp_m / sm_sum

            # Y gradient
            py_shift = py - py.max()
            exp_p = np.exp(py_shift / gamma)
            my_shift = -py + py.min()
            exp_m = np.exp(my_shift / gamma)
            sp_sum = exp_p.sum()
            sm_sum = exp_m.sum()
            gy = exp_p / sp_sum - exp_m / sm_sum

            # Scatter to macro gradients
            for k in range(n_pins):
                m = flat_macro[s + k]
                if m >= 0 and movable[m]:
                    grad[m, 0] += gx[k]
                    grad[m, 1] += gy[k]

        # --- Density gradient ---
        bins = np.zeros((n_bins, n_bins))
        macro_bin_x = np.clip((y_pos[:, 0] / bin_w).astype(int), 0, n_bins - 1)
        macro_bin_y = np.clip((y_pos[:, 1] / bin_h).astype(int), 0, n_bins - 1)
        for i in range(n_hard):
            bins[macro_bin_x[i], macro_bin_y[i]] += macro_area[i]

        bin_area = bin_w * bin_h
        for i in movable_idx:
            bx, by = macro_bin_x[i], macro_bin_y[i]
            overflow = bins[bx, by] / bin_area - target_density
            if overflow > 0:
                cx = (bx + 0.5) * bin_w
                cy = (by + 0.5) * bin_h
                dx = y_pos[i, 0] - cx
                dy = y_pos[i, 1] - cy
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                grad[i, 0] += density_weight * overflow * dx / dist
                grad[i, 1] += density_weight * overflow * dy / dist

        # Nesterov update
        velocity = momentum * velocity - lr * grad
        pos = pos + velocity

        # Clip to canvas
        for i in movable_idx:
            pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

        # Fix immovable
        for i in range(n_hard):
            if not movable[i]:
                pos[i] = benchmark.macro_positions[i].numpy().astype(np.float64)

        # Increase density weight periodically
        if iteration % 200 == 199:
            density_weight = min(density_weight * 2, 5.0)
            lr *= 0.9

    return pos


# ---------------------------------------------------------------------------
# Push-apart pre-processing
# ---------------------------------------------------------------------------

def _push_apart(pos, benchmark, max_iters=500, damping=0.6):
    """Iteratively push overlapping macros apart with minimum displacement."""
    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    pos = pos.copy()
    gap = 0.1

    for _it in range(max_iters):
        pairs = []
        for i in range(n_hard):
            for j in range(i + 1, n_hard):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                sx = (sizes[i, 0] + sizes[j, 0]) / 2 + gap
                sy = (sizes[i, 1] + sizes[j, 1]) / 2 + gap
                ovlp_x = sx - abs(dx)
                ovlp_y = sy - abs(dy)
                if ovlp_x > 0 and ovlp_y > 0:
                    pairs.append((ovlp_x * ovlp_y, i, j, dx, dy, sx, sy))

        if not pairs:
            break

        pairs.sort(reverse=True)

        for _, i, j, _dx, _dy, sx, sy in pairs:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            ovlp_x = sx - abs(dx)
            ovlp_y = sy - abs(dy)
            if ovlp_x <= 0 or ovlp_y <= 0:
                continue

            ai = sizes[i, 0] * sizes[i, 1]
            aj = sizes[j, 0] * sizes[j, 1]
            wi = (aj / (ai + aj)) if movable[i] else 0
            wj = (ai / (ai + aj)) if movable[j] else 0
            if wi == 0 and wj == 0:
                continue
            wt = wi + wj
            wi /= wt
            wj /= wt

            if ovlp_x < ovlp_y:
                push = ovlp_x * damping + 0.02
                s = 1.0 if dx >= 0 else -1.0
                pos[i, 0] -= s * push * wi
                pos[j, 0] += s * push * wj
            else:
                push = ovlp_y * damping + 0.02
                s = 1.0 if dy >= 0 else -1.0
                pos[i, 1] -= s * push * wi
                pos[j, 1] += s * push * wj

        for i in range(n_hard):
            if movable[i]:
                pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

    return pos


# ---------------------------------------------------------------------------
# Greedy legalization
# ---------------------------------------------------------------------------

def _legalize(pos, benchmark, order_type=0, step_mult=0.2):
    n_hard = benchmark.num_hard_macros
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    if order_type == 0:
        order = sorted(range(n_hard), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    elif order_type == 1:
        order = sorted(range(n_hard), key=lambda i: sizes[i, 0] * sizes[i, 1])
    else:
        order = list(range(n_hard))
        random.Random(order_type * 1337).shuffle(order)

    placed = np.zeros(n_hard, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue
        if placed.any():
            ddx = np.abs(legal[idx, 0] - legal[:, 0])
            ddy = np.abs(legal[idx, 1] - legal[:, 1])
            c = (ddx < sep_x[idx] + 0.05) & (ddy < sep_y[idx] + 0.05) & placed
            c[idx] = False
            if not c.any():
                placed[idx] = True
                continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * step_mult
        best_p, best_d = legal[idx].copy(), float("inf")
        for r in range(1, 300):
            ring_found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])
                    if placed.any():
                        ddx2 = np.abs(cx - legal[:, 0])
                        ddy2 = np.abs(cy - legal[:, 1])
                        c2 = (ddx2 < sep_x[idx] + 0.05) & (ddy2 < sep_y[idx] + 0.05) & placed
                        c2[idx] = False
                        if c2.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        ring_found = True
            if ring_found:
                break
        legal[idx] = best_p
        placed[idx] = True
    return legal


# ---------------------------------------------------------------------------
# Refine toward initial
# ---------------------------------------------------------------------------

def _refine_toward_initial(legal_pos, init_pos, benchmark, n_passes=20):
    n_hard = benchmark.num_hard_macros
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    pos = legal_pos.copy()
    gap = 0.05

    for _pass in range(n_passes):
        improved = False
        order = list(range(n_hard))
        random.shuffle(order)
        for i in order:
            if not movable[i]:
                continue
            dx = init_pos[i, 0] - pos[i, 0]
            dy = init_pos[i, 1] - pos[i, 1]
            if abs(dx) < 0.01 and abs(dy) < 0.01:
                continue
            for alpha in [0.8, 0.5, 0.3, 0.2, 0.1, 0.05]:
                nx = np.clip(pos[i, 0] + alpha * dx, half_w[i], cw - half_w[i])
                ny = np.clip(pos[i, 1] + alpha * dy, half_h[i], ch - half_h[i])
                ddx = np.abs(nx - pos[:, 0])
                ddy = np.abs(ny - pos[:, 1])
                conflicts = (ddx < sep_x[i] + gap) & (ddy < sep_y[i] + gap)
                conflicts[i] = False
                if not conflicts.any():
                    pos[i, 0] = nx
                    pos[i, 1] = ny
                    improved = True
                    break
        if not improved:
            break
    return pos


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------

def _macro_connectivity(benchmark):
    """Count number of nets each hard macro participates in (connectivity degree)."""
    n_hard = benchmark.num_hard_macros
    net_count = np.zeros(n_hard, dtype=np.int32)
    for net_nodes in benchmark.net_nodes:
        nodes = net_nodes.numpy() if hasattr(net_nodes, 'numpy') else np.array(net_nodes)
        for n in nodes:
            if 0 <= n < n_hard:
                net_count[n] += 1
    return net_count


def _macro_adjacency(benchmark):
    """Build adjacency list between hard macros that share nets."""
    n_hard = benchmark.num_hard_macros
    adj = [set() for _ in range(n_hard)]
    for net_nodes in benchmark.net_nodes:
        nodes = net_nodes.numpy() if hasattr(net_nodes, 'numpy') else np.array(net_nodes)
        hard_in_net = [int(n) for n in nodes if 0 <= n < n_hard]
        for a in hard_in_net:
            for b in hard_in_net:
                if a != b:
                    adj[a].add(b)
    return adj


# ---------------------------------------------------------------------------
# Coordinate descent with actual proxy cost
# ---------------------------------------------------------------------------

def _coord_descent(pos_np, benchmark, plc_eval, max_time=3000, incr_eval=None):
    """Coordinate descent with incremental evaluator (300x faster than official).

    If incr_eval is provided, uses it for all evaluations (verified to < 1e-6 accuracy).
    Otherwise falls back to official plc evaluator.

    Strategies:
      - Connectivity-aware ordering: most-connected macros first (highest impact)
      - First-improving for large deltas (>= 0.35): accept first improving
        direction, shuffle directions each time to avoid bias
      - Best-of-all-8 for small deltas (< 0.35): exhaustive direction search
      - Macro-size-scaled deltas: larger macros take proportionally larger steps
      - Multi-pass: loop through full delta schedule until no improvement or time
      - Reserve 15% of time for swap phase
    """
    from macro_place.objective import compute_proxy_cost

    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    # Connectivity-aware ordering: most connected macros first.
    # Use incr_eval.macro_nets if available (benchmark.net_nodes is empty in .pt files).
    if incr_eval is not None:
        net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_hard)])
    else:
        net_count = _macro_connectivity(benchmark)
    movable_sorted = sorted(movable_idx.tolist(), key=lambda i: -net_count[i])

    # Macro-size scaling factors: delta * scale[i] for each macro
    macro_max_dim = np.maximum(sizes[:, 0], sizes[:, 1])
    avg_macro_size = macro_max_dim[movable_idx].mean() if len(movable_idx) > 0 else 1.0
    size_scale = macro_max_dim / max(avg_macro_size, 1e-6)

    pos = pos_np.copy()
    gap = 0.05

    # Threshold: deltas above this use first-improving, below use best-of-all
    FIRST_IMPROVE_THRESHOLD = 0.35

    # Reserve 15% of time for swap phase
    cd_time_limit = max_time * 0.85
    swap_time_limit = max_time * 0.95

    def _check_overlap(idx):
        ddx = np.abs(pos[idx, 0] - pos[:, 0])
        ddy = np.abs(pos[idx, 1] - pos[:, 1])
        o = (ddx < sep_x[idx] + gap) & (ddy < sep_y[idx] + gap)
        o[idx] = False
        return o.any()

    # Use incremental evaluator if provided (300x faster)
    if incr_eval is not None:
        incr_eval.sync_positions(pos)

    def _eval_cost():
        if incr_eval is not None:
            return incr_eval.get_proxy_cost()
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(pos, dtype=torch.float32)
        return compute_proxy_cost(full, benchmark, plc_eval)["proxy_cost"]

    def _move_and_eval(idx, nx, ny):
        """Move macro idx and return new cost. Uses incremental if available."""
        if incr_eval is not None:
            return incr_eval.move_macro(idx, nx, ny)
        pos[idx, 0] = nx
        pos[idx, 1] = ny
        return _eval_cost()

    def _undo_move_eval(idx, old_x, old_y):
        """Undo move. Uses incremental if available."""
        if incr_eval is not None:
            incr_eval.undo_move()
        pos[idx, 0] = old_x
        pos[idx, 1] = old_y

    def _accept_move(idx, nx, ny):
        """Accept move (update pos array if using incremental, it's already applied)."""
        pos[idx, 0] = nx
        pos[idx, 1] = ny

    current_cost = _eval_cost()
    best_pos = pos.copy()
    best_cost = current_cost
    t0 = time.time()

    active = set(movable_idx.tolist())

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]

    # Finer deltas with incremental evaluator.
    # Larger initial deltas allow longer-range moves; fine deltas enable precision refinement.
    # size_scale amplifies for large macros, so effective range is delta * size_scale * macro_max_dim.
    delta_schedule = [5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02]

    # Multi-pass: loop through the full delta schedule repeatedly
    # With 300x incremental eval, try ALL macros every pass (no active set pruning)
    pass_num = 0
    while time.time() - t0 < cd_time_limit:
        pass_num += 1
        pass_improved = False

        for delta in delta_schedule:
            if time.time() - t0 > cd_time_limit:
                break

            next_active = set()
            order = list(movable_sorted)

            for i in order:
                if time.time() - t0 > cd_time_limit:
                    break

                # Scale delta by macro size
                scaled_delta = delta * size_scale[i]

                old_x, old_y = pos[i, 0], pos[i, 1]

                # With 300x incremental eval, always use best-of-all-8
                best_dir_cost = current_cost
                best_dir_pos = None

                for ddx, ddy in dirs:
                    nx = np.clip(old_x + scaled_delta * ddx, half_w[i], cw - half_w[i])
                    ny = np.clip(old_y + scaled_delta * ddy, half_h[i], ch - half_h[i])
                    if abs(nx - old_x) < 0.001 and abs(ny - old_y) < 0.001:
                        continue

                    pos[i, 0] = nx
                    pos[i, 1] = ny

                    if _check_overlap(i):
                        pos[i, 0] = old_x
                        pos[i, 1] = old_y
                        continue

                    cost = _move_and_eval(i, nx, ny)
                    if cost < best_dir_cost:
                        best_dir_cost = cost
                        best_dir_pos = (nx, ny)

                    _undo_move_eval(i, old_x, old_y)

                if best_dir_pos is not None:
                    _move_and_eval(i, best_dir_pos[0], best_dir_pos[1])
                    _accept_move(i, best_dir_pos[0], best_dir_pos[1])
                    current_cost = best_dir_cost
                    next_active.add(i)
                    pass_improved = True
                    if best_dir_cost < best_cost:
                        best_cost = best_dir_cost
                        best_pos = pos.copy()

        # If no improvement in this full pass, stop looping
        if not pass_improved:
            break

    # --- Swap phase with remaining 15% of time ---
    # Uses incremental evaluator for correct evaluation (Bug 1 fix).
    # Ordering: net-adjacent pairs first (most likely to improve), then by distance.
    if time.time() - t0 < swap_time_limit:
        pos = best_pos.copy()
        if incr_eval is not None:
            incr_eval.sync_positions(pos)
        current_cost = best_cost

        movable_set = set(movable_idx.tolist())

        # Build net-adjacent pairs: pairs of movable macros sharing a net
        net_adj_pairs = set()
        if incr_eval is not None:
            for i in movable_set:
                for nid in incr_eval.macro_nets[i]:
                    for j in incr_eval.net_macros[nid]:
                        if j > i and j in movable_set:
                            net_adj_pairs.add((i, j))
        net_adj_list = sorted(net_adj_pairs,
                              key=lambda p: -(net_count[p[0]] + net_count[p[1]]))

        # All movable pairs sorted by distance as fallback
        movable_list = sorted(movable_set, key=lambda i: -net_count[i])

        def _try_swap(i, j):
            """Try swapping macros i and j. Returns True if accepted."""
            nonlocal current_cost, best_cost, best_pos

            oi_x, oi_y = float(pos[i, 0]), float(pos[i, 1])
            oj_x, oj_y = float(pos[j, 0]), float(pos[j, 1])

            # Candidate positions after swap (clip to canvas)
            nix = float(np.clip(oj_x, half_w[i], cw - half_w[i]))
            niy = float(np.clip(oj_y, half_h[i], ch - half_h[i]))
            njx = float(np.clip(oi_x, half_w[j], cw - half_w[j]))
            njy = float(np.clip(oi_y, half_h[j], ch - half_h[j]))

            if incr_eval is not None:
                # Tentatively update pos for overlap checking
                pos[i, 0] = nix; pos[i, 1] = niy
                pos[j, 0] = njx; pos[j, 1] = njy

                # Check BOTH i and j for overlaps against all macros
                reject = False
                # Check i at nix,niy vs all macros (skip i and j)
                ddx_i = np.abs(nix - pos[:, 0])
                ddy_i = np.abs(niy - pos[:, 1])
                ovlp_i = (ddx_i < sep_x[i] + gap) & (ddy_i < sep_y[i] + gap)
                ovlp_i[i] = False
                ovlp_i[j] = False  # check i vs j separately
                if ovlp_i.any():
                    reject = True

                if not reject:
                    # Check j at njx,njy vs all macros (skip i and j)
                    ddx_j = np.abs(njx - pos[:, 0])
                    ddy_j = np.abs(njy - pos[:, 1])
                    ovlp_j = (ddx_j < sep_x[j] + gap) & (ddy_j < sep_y[j] + gap)
                    ovlp_j[i] = False
                    ovlp_j[j] = False
                    if ovlp_j.any():
                        reject = True

                if not reject:
                    # Check i vs j
                    if (abs(nix - njx) < sep_x[i, j] + gap and
                        abs(niy - njy) < sep_y[i, j] + gap):
                        reject = True

                if reject:
                    pos[i, 0] = oi_x; pos[i, 1] = oi_y
                    pos[j, 0] = oj_x; pos[j, 1] = oj_y
                    return False

                # Overlap-free — apply moves in incr_eval
                incr_eval.move_macro(i, nix, niy)
                swap_cost = incr_eval.move_macro(j, njx, njy)

                if swap_cost < current_cost:
                    current_cost = swap_cost
                    if swap_cost < best_cost:
                        best_cost = swap_cost
                        best_pos = pos.copy()
                    return True
                else:
                    # Reject: undo j then undo i
                    incr_eval.undo_move()
                    pos[j, 0] = oj_x; pos[j, 1] = oj_y
                    # Undo i by moving it back
                    incr_eval.move_macro(i, oi_x, oi_y)
                    # This forward-move-as-undo is slightly expensive but correct.
                    # Don't call undo_move() again — just leave incr_eval state consistent.
                    pos[i, 0] = oi_x; pos[i, 1] = oi_y
                    return False
            else:
                # Non-incremental path (fallback)
                pos[i, 0] = nix; pos[i, 1] = niy
                pos[j, 0] = njx; pos[j, 1] = njy
                if _check_overlap(i) or _check_overlap(j):
                    pos[i, 0] = oi_x; pos[i, 1] = oi_y
                    pos[j, 0] = oj_x; pos[j, 1] = oj_y
                    return False
                cost = _eval_cost()
                if cost < current_cost:
                    current_cost = cost
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos.copy()
                    return True
                else:
                    pos[i, 0] = oi_x; pos[i, 1] = oi_y
                    pos[j, 0] = oj_x; pos[j, 1] = oj_y
                    return False

        # Pass 1: net-adjacent pairs — loop until no improvement or time runs out
        while time.time() - t0 < max_time:
            improved = False
            for i, j in net_adj_list:
                if time.time() - t0 > max_time:
                    break
                if _try_swap(i, j):
                    improved = True
            if not improved:
                break

        # Pass 2: all movable pairs sorted by distance if time remains
        if time.time() - t0 < max_time:
            for ii, i in enumerate(movable_list):
                if time.time() - t0 > max_time:
                    break
                for j in movable_list[ii + 1:]:
                    if time.time() - t0 > max_time:
                        break
                    if (i, j) in net_adj_pairs or (j, i) in net_adj_pairs:
                        continue  # already covered in pass 1
                    _try_swap(i, j)

    return best_pos, best_cost


# ---------------------------------------------------------------------------
# Gradient descent with exact proxy cost (finite-difference gradients)
# ---------------------------------------------------------------------------

def _gradient_descent_exact(pos_np, benchmark, incr_eval, max_time):
    """Gradient descent using finite-difference gradients of the real proxy cost.

    For each movable macro, perturb x and y by ±epsilon and evaluate via
    incr_eval to get the true gradient direction (including congestion).
    Apply all gradient updates simultaneously, resolve overlaps with a
    lightweight push-apart, then sync to get the new cost.
    """
    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]
    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    if len(movable_idx) == 0:
        return pos_np.copy(), incr_eval.get_proxy_cost()

    pos = pos_np.copy()
    incr_eval.sync_positions(pos)
    current_cost = incr_eval.get_proxy_cost()
    best_pos = pos.copy()
    best_cost = current_cost

    epsilon = 0.05
    lr = 0.5
    decay = 0.95
    t0 = time.time()
    n_iter = 0

    while time.time() - t0 < max_time and lr >= 0.001:
        n_iter += 1

        # --- Compute gradient via central finite differences ---
        grad = np.zeros((n_hard, 2), dtype=np.float64)

        for i in movable_idx:
            if time.time() - t0 > max_time:
                break
            ox, oy = float(pos[i, 0]), float(pos[i, 1])

            # d/dx
            nx_p = min(ox + epsilon, cw - half_w[i])
            nx_m = max(ox - epsilon, half_w[i])
            dx_span = nx_p - nx_m
            if dx_span > 1e-6:
                c_p = incr_eval.move_macro(i, nx_p, oy)
                incr_eval.undo_move()
                c_m = incr_eval.move_macro(i, nx_m, oy)
                incr_eval.undo_move()
                grad[i, 0] = (c_p - c_m) / dx_span

            # d/dy
            ny_p = min(oy + epsilon, ch - half_h[i])
            ny_m = max(oy - epsilon, half_h[i])
            dy_span = ny_p - ny_m
            if dy_span > 1e-6:
                c_p = incr_eval.move_macro(i, ox, ny_p)
                incr_eval.undo_move()
                c_m = incr_eval.move_macro(i, ox, ny_m)
                incr_eval.undo_move()
                grad[i, 1] = (c_p - c_m) / dy_span

        if time.time() - t0 > max_time:
            break

        grad_norm = np.linalg.norm(grad[movable_idx])
        if grad_norm < 1e-10:
            break

        # --- Apply gradient to all movable macros simultaneously ---
        old_pos = pos.copy()
        pos[movable_idx] -= lr * grad[movable_idx]

        # Clip to canvas bounds
        pos[movable_idx, 0] = np.clip(pos[movable_idx, 0],
                                       half_w[movable_idx], cw - half_w[movable_idx])
        pos[movable_idx, 1] = np.clip(pos[movable_idx, 1],
                                       half_h[movable_idx], ch - half_h[movable_idx])

        # --- Lightweight push-apart (5 iterations max) ---
        gap = 0.05
        for _ in range(5):
            any_overlap = False
            for i in range(n_hard):
                for j in range(i + 1, n_hard):
                    if not movable[i] and not movable[j]:
                        continue
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    sx = sep_x_mat[i, j] + gap
                    sy = sep_y_mat[i, j] + gap
                    ovlp_x = sx - abs(dx)
                    ovlp_y = sy - abs(dy)
                    if ovlp_x > 0 and ovlp_y > 0:
                        any_overlap = True
                        mi = 0.5 if movable[i] else 0.0
                        mj = 0.5 if movable[j] else 0.0
                        if mi == 0.0 and mj == 0.0:
                            continue
                        total = mi + mj
                        wi, wj = mi / total, mj / total
                        if ovlp_x < ovlp_y:
                            push = ovlp_x * 0.6 + 0.01
                            s = 1.0 if dx >= 0 else -1.0
                            pos[i, 0] -= s * push * wi
                            pos[j, 0] += s * push * wj
                        else:
                            push = ovlp_y * 0.6 + 0.01
                            s = 1.0 if dy >= 0 else -1.0
                            pos[i, 1] -= s * push * wi
                            pos[j, 1] += s * push * wj
            # Re-clip after push
            pos[movable_idx, 0] = np.clip(pos[movable_idx, 0],
                                           half_w[movable_idx], cw - half_w[movable_idx])
            pos[movable_idx, 1] = np.clip(pos[movable_idx, 1],
                                           half_h[movable_idx], ch - half_h[movable_idx])
            if not any_overlap:
                break

        # --- Check for remaining overlaps; reject step if any ---
        has_overlap = False
        for i in range(n_hard):
            if has_overlap:
                break
            for j in range(i + 1, n_hard):
                if (abs(pos[i, 0] - pos[j, 0]) < sep_x_mat[i, j] + 0.05 and
                    abs(pos[i, 1] - pos[j, 1]) < sep_y_mat[i, j] + 0.05):
                    has_overlap = True
                    break

        if has_overlap:
            # Push-apart didn't resolve overlaps — revert and halve lr
            pos = old_pos.copy()
            incr_eval.sync_positions(pos)
            lr *= 0.5
            continue

        # --- Evaluate new cost ---
        incr_eval.sync_positions(pos)
        new_cost = incr_eval.get_proxy_cost()

        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos = pos.copy()
            lr *= decay
        else:
            # Revert and halve learning rate
            pos = old_pos.copy()
            incr_eval.sync_positions(pos)
            lr *= 0.5

    return best_pos, best_cost


# ---------------------------------------------------------------------------
# Simulated annealing with exact proxy cost
# ---------------------------------------------------------------------------

def _simulated_annealing(best_pos, best_cost, benchmark, incr_eval, max_time, t0):
    """Simulated annealing using incremental evaluator for exact proxy cost.

    Single-macro moves with exponential cooling schedule. Tracks best-ever
    and periodically resets to it if current solution drifts too far.
    """
    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    gap = 0.05

    if len(movable_idx) == 0:
        return best_pos.copy(), best_cost

    pos = best_pos.copy()
    incr_eval.sync_positions(pos)
    current_cost = incr_eval.get_proxy_cost()

    best_ever_pos = pos.copy()
    best_ever_cost = current_cost

    # Temperature schedule
    T_start = 0.05 * current_cost
    total_time = max_time
    sa_t0 = time.time()

    rng = np.random.RandomState(int(sa_t0 * 1000) % (2**31))

    # Precompute macro max dimensions for move scaling
    macro_max_dim = np.maximum(sizes[:, 0], sizes[:, 1])

    n_iter = 0
    n_accept = 0
    n_reject = 0
    n_skip = 0  # overlap skips

    while True:
        elapsed = time.time() - sa_t0
        if elapsed >= total_time:
            break

        n_iter += 1
        frac = elapsed / total_time  # 0 → 1

        # Temperature: exponential decay
        T = T_start * math.exp(-5.0 * frac)

        # Move scale: linear decay from 1.0 to 0.05
        temp_scale = 1.0 - 0.95 * frac

        # Pick random movable macro
        idx = movable_idx[rng.randint(len(movable_idx))]

        # Generate move from normal distribution
        std = macro_max_dim[idx] * temp_scale
        dx = rng.normal(0, std)
        dy = rng.normal(0, std)

        old_x, old_y = float(pos[idx, 0]), float(pos[idx, 1])
        nx = float(np.clip(old_x + dx, half_w[idx], cw - half_w[idx]))
        ny = float(np.clip(old_y + dy, half_h[idx], ch - half_h[idx]))

        if abs(nx - old_x) < 0.001 and abs(ny - old_y) < 0.001:
            continue

        # Check overlap
        ddx = np.abs(nx - pos[:, 0])
        ddy = np.abs(ny - pos[:, 1])
        ovlp = (ddx < sep_x[idx] + gap) & (ddy < sep_y[idx] + gap)
        ovlp[idx] = False
        if ovlp.any():
            n_skip += 1
            continue

        # Evaluate move
        new_cost = incr_eval.move_macro(idx, nx, ny)
        delta = new_cost - current_cost

        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            # Accept
            pos[idx, 0] = nx
            pos[idx, 1] = ny
            current_cost = new_cost
            n_accept += 1

            if current_cost < best_ever_cost:
                best_ever_cost = current_cost
                best_ever_pos = pos.copy()
        else:
            # Reject
            incr_eval.undo_move()
            n_reject += 1

        # Periodic reset to best-ever if drifted too far
        if n_iter % 10000 == 0:
            if current_cost > best_ever_cost * 1.03:
                pos = best_ever_pos.copy()
                incr_eval.sync_positions(pos)
                current_cost = incr_eval.get_proxy_cost()

        # Progress logging
        if n_iter % 50000 == 0:
            total_decisions = n_accept + n_reject
            accept_rate = n_accept / max(total_decisions, 1)
            print(f"  SA iter={n_iter:>8d}  cur={current_cost:.6f}  "
                  f"best={best_ever_cost:.6f}  T={T:.6f}  "
                  f"accept={accept_rate:.3f}  skip={n_skip}")

    # Final stats
    total_decisions = n_accept + n_reject
    accept_rate = n_accept / max(total_decisions, 1)
    print(f"  SA done: {n_iter} iters, best={best_ever_cost:.6f}, "
          f"accept_rate={accept_rate:.3f}, skipped={n_skip}")

    return best_ever_pos, best_ever_cost


# ---------------------------------------------------------------------------
# Perturbation phase: perturb + re-legalize to escape local optima
# ---------------------------------------------------------------------------

def _perturbation_phase(best_pos, best_cost, benchmark, init_pos, max_time, t0):
    """Repeatedly perturb the best placement and re-legalize to escape local optima.

    Two perturbation strategies:
      1. Random displacement: move 1-5 random macros by size-scaled offsets.
         If no overlaps created -> evaluate directly (fast path).
         If overlaps -> push_apart + legalize (2 orderings x 2 steps) + refine.
      2. Swap-based (every 3rd iteration): swap positions of connected macro pair,
         plus optional small perturbation on nearby macros.

    Accept if proxy cost improved (greedy). Final short CD burst if improvements
    were found.
    """
    from macro_place.objective import compute_proxy_cost

    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    if len(movable_idx) == 0:
        return best_pos, best_cost

    adj = _macro_adjacency(benchmark)
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    plc_eval = _load_plc(benchmark.name)
    cur_pos = best_pos.copy()
    cur_cost = best_cost
    iteration = 0
    n_accepted = 0

    def _has_any_overlap(pos_np):
        """Fast pairwise overlap check."""
        for i in range(n_hard):
            for j in range(i + 1, n_hard):
                if (abs(pos_np[i, 0] - pos_np[j, 0]) < sep_x[i, j] + 0.05 and
                    abs(pos_np[i, 1] - pos_np[j, 1]) < sep_y[i, j] + 0.05):
                    return True
        return False

    def _eval_cost(pos_np):
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
        result = compute_proxy_cost(full, benchmark, plc_eval)
        if result["overlap_count"] > 0:
            return float("inf")
        return result["proxy_cost"]

    while time.time() - t0 < max_time:
        iteration += 1

        # Alternate between displacement and swap perturbation
        if iteration % 3 != 0:
            # --- Strategy 1: Random displacement perturbation ---
            n_perturb = rng.randint(1, min(len(movable_idx), 5) + 1)
            chosen = rng.choice(movable_idx, size=n_perturb, replace=False)

            perturbed = cur_pos.copy()
            scale = rng.uniform(0.2, 1.2)
            for idx in chosen:
                max_dim = max(sizes[idx, 0], sizes[idx, 1])
                dx = rng.uniform(-1, 1) * max_dim * scale
                dy = rng.uniform(-1, 1) * max_dim * scale
                perturbed[idx, 0] = np.clip(perturbed[idx, 0] + dx,
                                            half_w[idx], cw - half_w[idx])
                perturbed[idx, 1] = np.clip(perturbed[idx, 1] + dy,
                                            half_h[idx], ch - half_h[idx])
        else:
            # --- Strategy 2: Swap-based perturbation (connected pairs) ---
            perturbed = cur_pos.copy()
            i = rng.choice(movable_idx)
            neighbors = [j for j in adj[i] if movable[j]]
            if not neighbors:
                # Fall back to nearest movable macro
                dists = np.abs(cur_pos[movable_idx, 0] - cur_pos[i, 0]) + \
                        np.abs(cur_pos[movable_idx, 1] - cur_pos[i, 1])
                dists[movable_idx == i] = float("inf")
                j = movable_idx[np.argmin(dists)]
            else:
                j = rng.choice(neighbors)

            perturbed[i, 0], perturbed[j, 0] = perturbed[j, 0], perturbed[i, 0]
            perturbed[i, 1], perturbed[j, 1] = perturbed[j, 1], perturbed[i, 1]
            perturbed[i, 0] = np.clip(perturbed[i, 0], half_w[i], cw - half_w[i])
            perturbed[i, 1] = np.clip(perturbed[i, 1], half_h[i], ch - half_h[i])
            perturbed[j, 0] = np.clip(perturbed[j, 0], half_w[j], cw - half_w[j])
            perturbed[j, 1] = np.clip(perturbed[j, 1], half_h[j], ch - half_h[j])

            # Small additional perturbation on 0-2 nearby macros
            n_extra = rng.randint(0, 3)
            if n_extra > 0:
                extra = rng.choice(movable_idx, size=min(n_extra, len(movable_idx)), replace=False)
                for idx in extra:
                    max_dim = max(sizes[idx, 0], sizes[idx, 1])
                    perturbed[idx, 0] = np.clip(perturbed[idx, 0] + rng.uniform(-0.3, 0.3) * max_dim,
                                                half_w[idx], cw - half_w[idx])
                    perturbed[idx, 1] = np.clip(perturbed[idx, 1] + rng.uniform(-0.3, 0.3) * max_dim,
                                                half_h[idx], ch - half_h[idx])

        if time.time() - t0 > max_time:
            break

        # --- Fast path: if no overlaps, evaluate directly ---
        if not _has_any_overlap(perturbed):
            cost = _eval_cost(perturbed)
            if cost < best_cost:
                best_cost = cost
                best_pos = perturbed.copy()
                cur_pos = perturbed.copy()
                cur_cost = cost
                n_accepted += 1
            continue

        # --- Slow path: push-apart + legalize (2 orderings x 2 steps) + refine ---
        pushed = _push_apart(perturbed, benchmark, max_iters=200, damping=0.6)

        cand_pos = None
        cand_cost = float("inf")
        ot_choices = [0, rng.randint(2, 20)]
        sm_choices = [0.08, 0.14]
        for ot in ot_choices:
            if time.time() - t0 > max_time:
                break
            for sm in sm_choices:
                if time.time() - t0 > max_time:
                    break
                legal = _legalize(pushed, benchmark, order_type=ot, step_mult=sm)
                refined = _refine_toward_initial(legal, init_pos, benchmark)
                if _has_any_overlap(refined):
                    continue
                cost = _eval_cost(refined)
                if cost < cand_cost:
                    cand_cost = cost
                    cand_pos = refined.copy()

        if cand_pos is not None and cand_cost < best_cost:
            best_cost = cand_cost
            best_pos = cand_pos.copy()
            cur_pos = cand_pos.copy()
            cur_cost = cand_cost
            n_accepted += 1

    # If we found improvements, do one final CD burst with remaining time
    if n_accepted > 0 and time.time() - t0 < max_time - 30:
        cd_time = min(120, max_time - (time.time() - t0) - 5)
        if cd_time > 10:
            plc_cd = _load_plc(benchmark.name)
            cd_pos, cd_cost = _coord_descent(best_pos, benchmark, plc_cd, max_time=cd_time)
            if cd_cost < best_cost:
                best_cost = cd_cost
                best_pos = cd_pos

    return best_pos, best_cost


# ---------------------------------------------------------------------------
# Parallel CD worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _cd_worker(args):
    """Run coordinate descent in a separate process with its own IncrementalEvaluator.

    Each worker loads its own plc/benchmark/incr_eval to avoid shared state.
    Returns (best_positions_bytes, best_cost, n_hard).
    """
    start_pos_bytes, n_hard_macros, benchmark_name, cd_time, seed = args
    start_pos = np.frombuffer(start_pos_bytes, dtype=np.float64).reshape(n_hard_macros, 2).copy()

    random.seed(seed)
    np.random.seed(seed)

    plc = _load_plc(benchmark_name)
    if plc is None:
        return start_pos.tobytes(), float('inf'), n_hard_macros

    from macro_place.benchmark import Benchmark
    bench = Benchmark.load(f'benchmarks/processed/public/{benchmark_name}.pt')
    incr_eval = IncrementalEvaluator(plc, bench)

    pos, cost = _coord_descent(start_pos, bench, plc, max_time=cd_time, incr_eval=incr_eval)
    return pos.tobytes(), cost, n_hard_macros


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class OptimalPlacer:
    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        plc = _load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()

        init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        t0 = time.time()

        from macro_place.objective import compute_proxy_cost

        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable)[0]

        # Note: quadratic, centroid, gradient, and analytical global placement
        # functions exist in this file but are not called. Tournament testing on
        # ibm01 showed they all lose by 20-24% to the conservative push-apart
        # variant, because the benchmark's initial placement is already nearly
        # optimal — local refinement beats global re-placement.

        # --- Phase 1: Push-apart pre-processing (3 configs) ---
        push_configs = [
            (300, 0.4),   # conservative
            (500, 0.6),   # moderate
            (800, 0.8),   # aggressive
        ]
        pushed_positions = []
        for max_it, damp in push_configs:
            pushed_positions.append(
                _push_apart(init_pos, benchmark, max_iters=max_it, damping=damp))

        # --- Phase 2: Multi-start legalization ---
        # Evaluate ALL candidates with actual proxy cost (no HPWL screening)
        # Time budget: 600s for legalization phase
        sizes_np = benchmark.macro_sizes[:n_hard].numpy()
        plc_eval = _load_plc(benchmark.name)
        best_pos = None
        best_cost = float("inf")
        n_evals = 0
        eval_time_sum = 0.0

        LEGALIZE_TIME_BUDGET = 600

        def _has_overlap(pos_np):
            for i in range(n_hard):
                for j in range(i + 1, n_hard):
                    if (abs(pos_np[i, 0] - pos_np[j, 0]) < (sizes_np[i, 0] + sizes_np[j, 0]) / 2 and
                        abs(pos_np[i, 1] - pos_np[j, 1]) < (sizes_np[i, 1] + sizes_np[j, 1]) / 2):
                        return True
            return False

        # Track best cost per starting-position type for diagnostics
        best_per_start = {}  # name -> (cost, count_tried)

        def _try_candidate(pos_np, start_name="?"):
            nonlocal best_pos, best_cost, n_evals, eval_time_sum
            if _has_overlap(pos_np):
                return
            full = benchmark.macro_positions.clone()
            full[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
            et0 = time.time()
            result = compute_proxy_cost(full, benchmark, plc_eval)
            eval_time_sum += time.time() - et0
            n_evals += 1
            if result["overlap_count"] == 0:
                cost = result["proxy_cost"]
                prev = best_per_start.get(start_name, (float("inf"), 0))
                best_per_start[start_name] = (min(prev[0], cost), prev[1] + 1)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos_np.copy()
                    nonlocal_winner[0] = start_name

        nonlocal_winner = ["?"]

        def _pos_hash(pos_np):
            quantized = np.round(pos_np * 10).astype(np.int32)
            return hash(quantized.tobytes())

        seen_hashes = set()
        step_sizes = [0.05, 0.08, 0.12, 0.18]
        n_orderings = 30

        # Build labeled list of starting positions
        labeled_starts = []
        for k, p in enumerate(pushed_positions):
            labeled_starts.append((p, f"push_{k}"))
        labeled_starts.append((init_pos, "raw"))

        # Round-robin: for each (ordering, step_size), try all starts
        for ot in range(n_orderings):
            if time.time() - t0 > LEGALIZE_TIME_BUDGET:
                break
            for sm in step_sizes:
                if time.time() - t0 > LEGALIZE_TIME_BUDGET:
                    break
                for start_pos, start_name in labeled_starts:
                    if time.time() - t0 > LEGALIZE_TIME_BUDGET:
                        break
                    legal = _legalize(start_pos, benchmark, order_type=ot, step_mult=sm)
                    refined = _refine_toward_initial(legal, init_pos, benchmark)
                    h = _pos_hash(refined)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    _try_candidate(refined, start_name)

        # Diagnostic prints
        for name in ("push_0", "push_1", "push_2", "raw"):
            if name in best_per_start:
                c, n = best_per_start[name]
                print(f"  Init '{name}' best legalized cost: {c:.6f} ({n} tried)")
        print(f"  Selected init: {nonlocal_winner[0]} (cost {best_cost:.6f})")

        legalize_time = time.time() - t0

        # --- Phase 3: Parallel multi-restart CD ---
        # 1. Run first CD + GD sequentially (establishes baseline)
        # 2. Generate perturbed starting positions
        # 3. Launch parallel CD workers on all available cores
        # 4. Take best result across all restarts
        if best_pos is not None:
            plc_cd = _load_plc(benchmark.name)
            incr_eval = IncrementalEvaluator(plc_cd, benchmark)
            TOTAL_TIME_LIMIT = 3300  # safety margin under 3600s competition limit

            # --- First CD run (sequential, with GD) ---
            first_cd_budget = max(200, min(2800, TOTAL_TIME_LIMIT - (time.time() - t0) - 200))
            cd_pos, cd_cost = _coord_descent(
                best_pos, benchmark, plc_cd, max_time=first_cd_budget, incr_eval=incr_eval)
            if cd_cost < best_cost:
                best_cost = cd_cost
                best_pos = cd_pos

            # GD on first result
            gd_remaining = max(0, TOTAL_TIME_LIMIT - (time.time() - t0) - 200)
            if gd_remaining > 30:
                gd_pos, gd_cost = _gradient_descent_exact(
                    best_pos, benchmark, incr_eval, max_time=min(300, gd_remaining))
                if gd_cost < best_cost:
                    best_cost = gd_cost
                    best_pos = gd_pos

            # --- Parallel restart phase ---
            parallel_remaining = TOTAL_TIME_LIMIT - (time.time() - t0)
            if parallel_remaining > 120:
                n_workers = min(15, max(1, mp.cpu_count() - 1))
                rng_restart = np.random.RandomState(42)

                # Generate perturbed starting positions
                worker_starts = []
                for w in range(n_workers):
                    cd_start = best_pos.copy()
                    n_perturb = max(2, min(len(movable_idx), rng_restart.randint(3, 8)))
                    chosen = rng_restart.choice(movable_idx, size=n_perturb, replace=False)
                    for idx in chosen:
                        max_dim = max(sizes_np[idx, 0], sizes_np[idx, 1])
                        scale = rng_restart.uniform(0.3, 1.5)
                        cd_start[idx, 0] = np.clip(
                            cd_start[idx, 0] + rng_restart.uniform(-1, 1) * max_dim * scale,
                            sizes_np[idx, 0] / 2, float(benchmark.canvas_width) - sizes_np[idx, 0] / 2)
                        cd_start[idx, 1] = np.clip(
                            cd_start[idx, 1] + rng_restart.uniform(-1, 1) * max_dim * scale,
                            sizes_np[idx, 1] / 2, float(benchmark.canvas_height) - sizes_np[idx, 1] / 2)

                    # Resolve overlaps
                    cd_start = _push_apart(cd_start, benchmark, max_iters=200, damping=0.6)
                    if _has_overlap(cd_start):
                        ot = rng_restart.randint(0, 20)
                        sm = rng_restart.choice([0.05, 0.08, 0.12, 0.18])
                        cd_start = _legalize(cd_start, benchmark, order_type=ot, step_mult=sm)
                        cd_start = _refine_toward_initial(cd_start, init_pos, benchmark)
                        if _has_overlap(cd_start):
                            continue  # skip invalid starts

                    worker_starts.append(cd_start)

                if worker_starts:
                    # Each worker gets the full remaining time
                    cd_time_per_worker = max(60, parallel_remaining - 30)
                    worker_args = [
                        (s.tobytes(), n_hard, benchmark.name, cd_time_per_worker, 1000 + w)
                        for w, s in enumerate(worker_starts)
                    ]

                    try:
                        with mp.Pool(len(worker_starts)) as pool:
                            results = pool.map(_cd_worker, worker_args)

                        for pos_bytes, cost, nh in results:
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = np.frombuffer(pos_bytes, dtype=np.float64).reshape(nh, 2).copy()
                    except Exception:
                        pass  # fall back to sequential result if parallel fails

        # Build final placement
        full_pos = benchmark.macro_positions.clone()
        if best_pos is not None:
            full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        return full_pos
