#!/usr/bin/env python3
"""
fes_analysis.py  —  Free Energy Surface analyzer

Finds local minima on a PLUMED 2D FES, assigns trajectory frames to basins,
and extracts per-basin XTC sub-trajectories.  Handles periodic CVs (torsion
angles, etc.) by tiling the grid so that basins wrap correctly across ±π
boundaries.

Requirements:
    pip install numpy scipy pandas scikit-image matplotlib MDAnalysis
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, minimum_filter, label as ndlabel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── FES loading ────────────────────────────────────────────────────────────────

def load_fes(filename):
    """
    Parse a PLUMED fes.dat file.

    Returns:
        cv1_axis    : 1D array, CV1 grid values
        cv2_axis    : 1D array, CV2 grid values
        fes_grid    : 2D array, shape (n_cv1, n_cv2)
        cv_names    : [cv1_name, cv2_name]
        periodic    : [bool_cv1, bool_cv2]  — True if that CV is periodic
    """
    cv_names = None
    periodic_flags: dict[str, bool] = {}
    rows = []

    with open(filename) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('#! FIELDS'):
                cv_names = s.split()[2:4]
            elif s.startswith('#! SET periodic_'):
                parts = s.split()        # ['#!', 'SET', 'periodic_cvname', 'true/false']
                key   = parts[2][len('periodic_'):]
                periodic_flags[key] = (parts[3].lower() == 'true')
            elif s.startswith('#') or s == '':
                continue
            else:
                rows.append([float(v) for v in s.split()])

    if cv_names is None:
        raise ValueError(f"No '#! FIELDS' header found in {filename}")

    data = np.array(rows)
    cv1_all, cv2_all, fes_all = data[:, 0], data[:, 1], data[:, 2]

    cv1_axis = np.unique(cv1_all)
    cv2_axis = np.unique(cv2_all)
    n1, n2 = len(cv1_axis), len(cv2_axis)

    if data.shape[0] != n1 * n2:
        raise ValueError(
            f"Non-rectangular grid in {filename}: "
            f"{data.shape[0]} data points but expected {n1}×{n2} = {n1*n2}"
        )

    # Detect which CV is the inner (fast-varying) loop by checking whether
    # cv1 or cv2 changes between the first two data rows.
    # After this, fes_grid[i, j] == FES(cv1_axis[i], cv2_axis[j]) always.
    if not np.isclose(cv1_all[0], cv1_all[1]):
        # cv1 is inner loop (varies fastest); cv2 is outer loop
        # Memory layout: [cv2_idx * n1 + cv1_idx]
        fes_grid = fes_all.reshape(n2, n1).T   # (n2,n1) → T → (n1,n2)
    else:
        # cv2 is inner loop; cv1 is outer loop
        # Memory layout: [cv1_idx * n2 + cv2_idx]
        fes_grid = fes_all.reshape(n1, n2)

    periodic = [
        periodic_flags.get(cv_names[0], False),
        periodic_flags.get(cv_names[1], False),
    ]
    return cv1_axis, cv2_axis, fes_grid, list(cv_names), periodic


# ── Smoothing ──────────────────────────────────────────────────────────────────

def smooth_fes(fes_grid, sigma, periodic):
    """
    Masked Gaussian smoothing.  Uses 'wrap' mode for periodic axes so that
    the smoothed surface is continuous across the ±π boundary.
    """
    if sigma <= 0:
        return fes_grid.copy()

    p1, p2 = periodic
    if p1 and p2:
        mode = 'wrap'
    elif not p1 and not p2:
        mode = 'reflect'
    else:
        mode = ('wrap' if p1 else 'reflect', 'wrap' if p2 else 'reflect')

    finite  = np.isfinite(fes_grid)
    data    = np.where(finite, fes_grid, 0.0)
    weights = finite.astype(float)

    blurred_data    = gaussian_filter(data,    sigma=sigma, mode=mode)
    blurred_weights = gaussian_filter(weights, sigma=sigma, mode=mode)

    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = np.where(blurred_weights > 1e-6,
                            blurred_data / blurred_weights,
                            np.inf)
    return smoothed


# ── Minima detection ───────────────────────────────────────────────────────────

def find_minima(smoothed, cv1_axis, cv2_axis, threshold_rel, neighborhood,
                periodic):
    """
    Detect local minima ≤ global_min + threshold_rel.

    For periodic CVs minimum_filter wraps around the boundary, so a minimum
    that straddles ±π is detected as a single point rather than two spurious
    copies at the grid edges.

    Returns:
        minima      : list of dicts {id, i, j, cv1, cv2, energy, delta_e}
        global_min  : float
    """
    finite = np.isfinite(smoothed)
    global_min = smoothed[finite].min()
    cutoff = global_min + threshold_rel

    p1, p2 = periodic
    if p1 and p2:
        mode = 'wrap'
    elif not p1 and not p2:
        mode = 'nearest'
    else:
        mode = ('wrap' if p1 else 'nearest', 'wrap' if p2 else 'nearest')

    padded = np.where(finite, smoothed, np.inf)
    local_min_val = minimum_filter(padded, size=neighborhood, mode=mode)
    is_min = (smoothed == local_min_val) & finite & (smoothed <= cutoff)

    # Merge adjacent minimum pixels — pick the deepest in each cluster.
    struct = np.ones((3, 3), dtype=int)
    labeled, n_clusters = ndlabel(is_min, structure=struct)

    minima = []
    for k in range(1, n_clusters + 1):
        pts = np.argwhere(labeled == k)
        best = pts[np.argmin(smoothed[pts[:, 0], pts[:, 1]])]
        i, j = int(best[0]), int(best[1])
        minima.append({
            'i': i, 'j': j,
            'cv1':     cv1_axis[i],
            'cv2':     cv2_axis[j],
            'energy':  float(smoothed[i, j]),
            'delta_e': float(smoothed[i, j] - global_min),
        })

    minima.sort(key=lambda m: m['energy'])
    for rank, m in enumerate(minima, 1):
        m['id'] = rank

    return minima, global_min


# ── Basin definition ───────────────────────────────────────────────────────────

def _tiled(smoothed, minima, periodic):
    """
    Return a (possibly tiled) copy of smoothed and marker/offset info.

    For periodic axes the grid is tiled 3× so that the watershed / flood-fill
    naturally wraps across the boundary.  We run the basin algorithm on the
    tiled surface, then extract the central tile.
    """
    p1, p2 = periodic
    n1, n2 = smoothed.shape
    t1 = 3 if p1 else 1
    t2 = 3 if p2 else 1

    tiled_fes = np.tile(smoothed, (t1, t2))

    # Canonical (central) tile offsets
    ci = (t1 // 2) * n1
    cj = (t2 // 2) * n2

    # Place a copy of every minimum in each tile
    markers = np.zeros(tiled_fes.shape, dtype=int)
    for m in minima:
        for di in range(t1):
            for dj in range(t2):
                markers[m['i'] + di * n1, m['j'] + dj * n2] = m['id']

    return tiled_fes, markers, (ci, cj, n1, n2)


def define_basins_watershed(smoothed, minima, threshold_rel, global_min, periodic):
    """
    Watershed segmentation: each grid point drains to exactly one minimum.
    Periodic boundaries are handled by tiling the grid before running watershed.
    """
    from skimage.segmentation import watershed

    tiled_fes, markers, (ci, cj, n1, n2) = _tiled(smoothed, minima, periodic)
    mask = np.isfinite(tiled_fes) & (tiled_fes <= global_min + threshold_rel)

    tiled_basins = watershed(tiled_fes, markers=markers, mask=mask)
    return tiled_basins[ci:ci + n1, cj:cj + n2]


def define_basins_contour(smoothed, minima, contour_width, periodic):
    """
    Flood-fill from each minimum up to contour_width kJ/mol above it.
    Lower-energy minima claim contested overlap.  Periodic boundaries are
    handled by tiling the grid.
    """
    struct = np.ones((3, 3), dtype=int)
    tiled_fes, _, (ci, cj, n1, n2) = _tiled(smoothed, minima, periodic)
    tbasins = np.zeros(tiled_fes.shape, dtype=int)

    for m in minima:
        cutoff    = m['energy'] + contour_width
        candidate = np.isfinite(tiled_fes) & (tiled_fes <= cutoff)
        labeled, _ = ndlabel(candidate, structure=struct)
        region = labeled[ci + m['i'], cj + m['j']]
        if region > 0:
            unowned = (labeled == region) & (tbasins == 0)
            tbasins[unowned] = m['id']

    return tbasins[ci:ci + n1, cj:cj + n2]


# ── Minimum energy paths ───────────────────────────────────────────────────────

def find_mep(smoothed, minima, periodic):
    """
    Find the minimum energy path (MEP) between every pair of basin minima.

    Uses Dijkstra's minimax algorithm: the path cost is the maximum energy
    encountered, so the optimal path crosses the lowest possible saddle point.

    Returns a list of dicts sorted by forward barrier height:
        basin_a, basin_b, saddle_energy,
        barrier_a_to_b, barrier_b_to_a,
        saddle_i, saddle_j,   — grid indices of saddle point
        path_ij               — list of (i, j) tuples
    """
    import heapq

    n1, n2 = smoothed.shape
    p1, p2 = periodic
    pad = np.where(np.isfinite(smoothed), smoothed, np.inf)

    def neighbors(i, j):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if p1:   ni = ni % n1
                elif not (0 <= ni < n1): continue
                if p2:   nj = nj % n2
                elif not (0 <= nj < n2): continue
                if np.isfinite(pad[ni, nj]):
                    yield ni, nj

    results = []
    for a in range(len(minima)):
        for b in range(a + 1, len(minima)):
            m1, m2 = minima[a], minima[b]
            start  = (m1['i'], m1['j'])
            goal   = (m2['i'], m2['j'])

            dist = np.full((n1, n2), np.inf)
            prev = {}
            dist[start] = float(pad[start])
            heap = [(float(pad[start]), start[0], start[1])]

            found = False
            while heap:
                d, i, j = heapq.heappop(heap)
                if (i, j) == goal:
                    found = True
                    break
                if d > dist[i, j]:
                    continue
                for ni, nj in neighbors(i, j):
                    nd = max(d, float(pad[ni, nj]))
                    if nd < dist[ni, nj]:
                        dist[ni, nj] = nd
                        prev[(ni, nj)] = (i, j)
                        heapq.heappush(heap, (nd, ni, nj))

            if not found:
                continue

            # Reconstruct path
            path, curr = [], goal
            while curr != start:
                path.append(curr)
                if curr not in prev:
                    break
                curr = prev[curr]
            path.append(start)
            path.reverse()

            # Saddle = highest-energy interior point on the path
            interior = path[1:-1]
            if interior:
                s_idx   = int(np.argmax([pad[i, j] for (i, j) in interior]))
                saddle_pt = interior[s_idx]
            else:
                saddle_pt = path[len(path) // 2]

            saddle_e = float(dist[goal])
            results.append(dict(
                basin_a=m1['id'],  basin_b=m2['id'],
                energy_a=m1['energy'], energy_b=m2['energy'],
                saddle_energy=saddle_e,
                barrier_a_to_b=saddle_e - m1['energy'],
                barrier_b_to_a=saddle_e - m2['energy'],
                saddle_i=saddle_pt[0], saddle_j=saddle_pt[1],
                path_ij=path,
            ))

    results.sort(key=lambda r: r['barrier_a_to_b'])
    return results


def _path_cv_coords(path_ij, cv1_axis, cv2_axis):
    """
    Convert (i,j) index path to (cv1, cv2) coordinate arrays.
    Inserts NaN where the path wraps across a periodic boundary (index jump > 1)
    so matplotlib draws a line break rather than a spurious cross-box segment.
    """
    if not path_ij:
        return np.array([]), np.array([])
    xs, ys = [], []
    pi, pj = path_ij[0]
    for ci, cj in path_ij:
        if abs(ci - pi) > 1 or abs(cj - pj) > 1:
            xs.append(np.nan); ys.append(np.nan)
        xs.append(cv1_axis[ci])
        ys.append(cv2_axis[cj])
        pi, pj = ci, cj
    return np.array(xs), np.array(ys)


def find_adjacent_pairs(basins, periodic):
    """
    Return the set of (a, b) basin-ID pairs (a < b) that share at least one
    grid-point boundary in the watershed basin map, respecting periodic boundaries.
    """
    n1, n2 = basins.shape
    p1, p2 = periodic
    pairs = set()
    for i in range(n1):
        for j in range(n2):
            a = int(basins[i, j])
            if a == 0:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % n1 if p1 else i + di
                    nj = (j + dj) % n2 if p2 else j + dj
                    if not (0 <= ni < n1 and 0 <= nj < n2):
                        continue
                    b = int(basins[ni, nj])
                    if b != 0 and b != a:
                        pairs.add((min(a, b), max(a, b)))
    return pairs


def plot_mep(smoothed, cv1_axis, cv2_axis, minima, mep_results,
             global_min, threshold_rel, cv_names, output_dir):
    """FES with all MEPs overlaid; saddle points marked with X."""
    import matplotlib.lines as mlines

    vmax_disp = global_min + threshold_rel + 5
    disp = np.where(np.isfinite(smoothed),
                    np.clip(smoothed, global_min, vmax_disp + 20),
                    vmax_disp + 20)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(cv1_axis, cv2_axis, disp.T, levels=30,
                     cmap='RdYlBu_r', vmin=global_min, vmax=vmax_disp, alpha=0.85)
    ax.contour(cv1_axis, cv2_axis, disp.T, levels=15,
               colors='k', linewidths=0.4, alpha=0.3)
    ax.contour(cv1_axis, cv2_axis, smoothed.T,
               levels=[global_min + threshold_rel],
               colors='white', linewidths=1.5, linestyles='--')
    plt.colorbar(cf, ax=ax, label='Free Energy (kJ/mol)')

    for m in minima:
        ax.plot(m['cv1'], m['cv2'], '*', ms=14,
                color='white', mec='black', mew=0.8, zorder=6)
        ax.text(m['cv1'], m['cv2'], f"  {m['id']}",
                fontsize=9, color='white', fontweight='bold', va='center', zorder=6)

    cmap_p = plt.cm.tab10
    handles = []
    for k, r in enumerate(mep_results):
        col = cmap_p(k % 10)
        xs, ys = _path_cv_coords(r['path_ij'], cv1_axis, cv2_axis)
        ax.plot(xs, ys, '-', color=col, lw=1.8, alpha=0.9, zorder=4)
        ax.plot(cv1_axis[r['saddle_i']], cv2_axis[r['saddle_j']],
                'X', ms=10, color=col, mec='black', mew=0.7, zorder=7)
        handles.append(mlines.Line2D(
            [0], [0], color=col, lw=2,
            label=(f"{r['basin_a']}↔{r['basin_b']}: "
                   f"{r['barrier_a_to_b']:.1f} / {r['barrier_b_to_a']:.1f} kJ/mol"),
        ))

    ax.legend(handles=handles, loc='upper right', fontsize=7,
              title='A→B / B→A barriers (kJ/mol)', title_fontsize=7,
              framealpha=0.7)
    ax.set_xlabel(cv_names[0])
    ax.set_ylabel(cv_names[1])
    ax.set_title('Minimum Energy Paths\n(✕ = saddle point,  ★ = basin minimum)')

    out = os.path.join(output_dir, 'mep_paths.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {out}")


def write_mep_summary(mep_results, minima, cv1_axis, cv2_axis, cv_names, output_dir):
    min_by_id = {m['id']: m for m in minima}
    rows = []
    for r in mep_results:
        ma, mb = min_by_id[r['basin_a']], min_by_id[r['basin_b']]
        rows.append({
            'basin_a':                r['basin_a'],
            'basin_b':                r['basin_b'],
            f'{cv_names[0]}_a':       round(ma['cv1'], 4),
            f'{cv_names[1]}_a':       round(ma['cv2'], 4),
            f'{cv_names[0]}_b':       round(mb['cv1'], 4),
            f'{cv_names[1]}_b':       round(mb['cv2'], 4),
            f'saddle_{cv_names[0]}':  round(cv1_axis[r['saddle_i']], 4),
            f'saddle_{cv_names[1]}':  round(cv2_axis[r['saddle_j']], 4),
            'saddle_energy_kJ_mol':   round(r['saddle_energy'],   3),
            'barrier_a_to_b_kJ_mol':  round(r['barrier_a_to_b'], 3),
            'barrier_b_to_a_kJ_mol':  round(r['barrier_b_to_a'], 3),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, 'mep_barriers.csv')
    df.to_csv(out, index=False)
    print(f"      Saved: {out}")
    return df


# ── COLVAR & trajectory ────────────────────────────────────────────────────────

def load_colvar(filename, cv_names):
    """
    Parse a PLUMED COLVAR file.

    Returns a DataFrame with at least columns 'time' and the two CV columns.
    """
    fields = None
    with open(filename) as fh:
        for line in fh:
            if line.startswith('#! FIELDS'):
                fields = line.split()[2:]
                break
    if fields is None:
        raise ValueError(f"No '#! FIELDS' header found in {filename}")

    for cv in cv_names:
        if cv not in fields:
            raise ValueError(
                f"CV '{cv}' (from FES) not in COLVAR FIELDS: {fields}\n"
                "Use --cv1 / --cv2 to override the CV column names."
            )
    if 'time' not in fields:
        raise ValueError(
            f"No 'time' column in COLVAR FIELDS: {fields}\n"
            "Time is required to match COLVAR frames to XTC frames."
        )

    df = pd.read_csv(
        filename, comment='#', sep=r'\s+',
        header=None, names=fields, engine='python'
    )
    return df


def match_xtc_to_colvar(colvar_df, xtc_file, tpr_file):
    """
    For each XTC frame find the single nearest COLVAR row by simulation time.
    Each XTC frame is assigned to exactly one COLVAR row (1:1).

    Returns:
        universe          : MDAnalysis Universe
        xtc_to_colvar_idx : int array shape (n_xtc,) — COLVAR row index per XTC frame
        max_dt            : float — worst-case time gap (for sanity check)
    """
    import MDAnalysis as mda

    print(f"      Reading XTC frame times from {xtc_file} …")
    u = mda.Universe(tpr_file, xtc_file)
    xtc_times = np.array([ts.time for ts in u.trajectory])
    print(f"      {len(xtc_times)} XTC frames found")

    colvar_times = colvar_df['time'].values

    # For each XTC time, find the nearest COLVAR row (O(n log m))
    pos       = np.searchsorted(colvar_times, xtc_times).clip(0, len(colvar_times) - 1)
    pos_below = (pos - 1).clip(0, len(colvar_times) - 1)
    dist_at    = np.abs(xtc_times - colvar_times[pos])
    dist_below = np.abs(xtc_times - colvar_times[pos_below])
    nearest    = np.where(dist_below < dist_at, pos_below, pos)
    max_dt     = float(np.minimum(dist_at, dist_below).max())

    return u, nearest, max_dt


def assign_frames_to_basins(colvar_df, basins, cv1_axis, cv2_axis, cv_names,
                             smoothed=None):
    """
    Nearest-grid-point lookup: map each COLVAR frame's (CV1, CV2) to a basin
    label.  Adds a 'basin' column (0 = outside all basins) and, if smoothed is
    provided, a 'fes_energy' column with the FES value at that grid point.
    """
    def nearest_idx(vals, axis):
        pos       = np.searchsorted(axis, vals).clip(0, len(axis) - 1)
        pos_below = (pos - 1).clip(0, len(axis) - 1)
        closer    = np.abs(vals - axis[pos_below]) < np.abs(vals - axis[pos])
        pos[closer] = pos_below[closer]
        return pos

    i_idx = nearest_idx(colvar_df[cv_names[0]].values, cv1_axis)
    j_idx = nearest_idx(colvar_df[cv_names[1]].values, cv2_axis)

    colvar_df = colvar_df.copy()
    colvar_df['basin'] = basins[i_idx, j_idx]
    if smoothed is not None:
        colvar_df['fes_energy'] = smoothed[i_idx, j_idx]
    return colvar_df


# ── Output writers ─────────────────────────────────────────────────────────────

def extract_xtc_frames(universe, frame_indices, output_path, center_ag=None):
    """
    Write selected XTC frames to output_path.

    If center_ag is provided, each frame is centered on that AtomGroup:
      1. unwrap(compound='fragments') — make the group whole across PBC
      2. translate all atoms so center_ag center-of-geometry is at box centre
      3. wrap(compound='fragments') — fold everything back in, keeping molecules whole
    """
    import MDAnalysis as mda

    valid = sorted({f for f in frame_indices if f >= 0})
    if not valid:
        return 0
    all_atoms = universe.select_atoms('all')
    with mda.Writer(output_path, all_atoms.n_atoms) as W:
        for fi in valid:
            ts = universe.trajectory[fi]
            if center_ag is not None:
                center_ag.unwrap(compound='fragments')
                box_center = ts.dimensions[:3] / 2.0
                shift = box_center - center_ag.center_of_geometry()
                all_atoms.translate(shift)
                all_atoms.wrap(compound='fragments')
            W.write(all_atoms)
    return len(valid)


def write_frame_list(basin, basin_df, cv_names, output_dir):
    path = os.path.join(output_dir, f'basin_{basin["id"]}_frames.txt')
    with open(path, 'w') as fh:
        fh.write(
            f"# Basin {basin['id']}:  "
            f"{cv_names[0]}={basin['cv1']:.4f}  "
            f"{cv_names[1]}={basin['cv2']:.4f}  "
            f"ΔE={basin['delta_e']:.3f} kJ/mol\n"
        )
        fh.write("# colvar_row  time\n")
        for idx, row in basin_df.iterrows():
            fh.write(f"{idx}  {row['time']:.6f}\n")
    return path


def write_summary(minima, colvar_df, output_dir, cv_names):
    total = len(colvar_df)
    rows  = []
    for m in minima:
        sub = colvar_df[colvar_df['basin'] == m['id']]
        n   = len(sub)
        rows.append({
            'basin_id':              m['id'],
            f'{cv_names[0]}_minimum': round(m['cv1'], 4),
            f'{cv_names[1]}_minimum': round(m['cv2'], 4),
            'energy_kJ_mol':         round(m['energy'],  3),
            'delta_E_kJ_mol':        round(m['delta_e'], 3),
            'n_frames':              n,
            'population_pct':        round(100 * n / total, 2) if total else float('nan'),
            f'mean_{cv_names[0]}':   round(sub[cv_names[0]].mean(), 4) if n else float('nan'),
            f'mean_{cv_names[1]}':   round(sub[cv_names[1]].mean(), 4) if n else float('nan'),
            f'std_{cv_names[0]}':    round(sub[cv_names[0]].std(),  4) if n else float('nan'),
            f'std_{cv_names[1]}':    round(sub[cv_names[1]].std(),  4) if n else float('nan'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'basin_summary.csv'), index=False)
    return df


# ── Plotting ───────────────────────────────────────────────────────────────────

def _get_cmap(name, n):
    """Matplotlib-version-agnostic colormap fetcher."""
    try:
        return matplotlib.colormaps[name].resampled(n)
    except AttributeError:
        return matplotlib.cm.get_cmap(name, n)  # type: ignore[attr-defined]


def plot_fes(smoothed, cv1_axis, cv2_axis, minima, basins,
             global_min, threshold_rel, cv_names, output_dir, method_tag=''):
    """
    Two-panel figure: (left) smoothed FES with threshold contour and minima
                      (right) basin assignment map.
    """
    n_basins  = len(minima)
    vmax_disp = global_min + threshold_rel + 5

    disp = np.where(
        np.isfinite(smoothed),
        np.clip(smoothed, global_min, vmax_disp + 20),
        vmax_disp + 20,
    )

    fig, (ax_fes, ax_bas) = plt.subplots(1, 2, figsize=(14, 6))

    # cv1 on x-axis, cv2 on y-axis.
    # contourf(x, y, Z) expects Z[row, col] = Z[y_idx, x_idx], so transpose.
    disp_T = disp.T        # shape (n_cv2, n_cv1)

    # Left: FES heatmap ───────────────────────────────────────────────────────
    cf = ax_fes.contourf(
        cv1_axis, cv2_axis, disp_T, levels=30,
        cmap='RdYlBu_r', vmin=global_min, vmax=vmax_disp,
    )
    ax_fes.contour(cv1_axis, cv2_axis, disp_T, levels=15,
                   colors='k', linewidths=0.4, alpha=0.4)
    ax_fes.contour(cv1_axis, cv2_axis, smoothed.T,
                   levels=[global_min + threshold_rel],
                   colors='white', linewidths=1.5, linestyles='--')
    for m in minima:
        ax_fes.plot(m['cv1'], m['cv2'], '*', ms=12,
                    color='white', mec='black', mew=0.8, zorder=5)
        ax_fes.text(m['cv1'], m['cv2'], f"  {m['id']}",
                    fontsize=8, color='white', va='center', zorder=5)
    plt.colorbar(cf, ax=ax_fes, label='Free Energy (kJ/mol)')
    ax_fes.set_xlabel(cv_names[0])
    ax_fes.set_ylabel(cv_names[1])
    ax_fes.set_title('Smoothed FES\n(white dashed = threshold,  ★ = minima)')

    # Right: basin map ────────────────────────────────────────────────────────
    cmap_b     = _get_cmap('tab20', n_basins + 1)
    basin_disp = np.where(basins > 0, basins.astype(float), np.nan)
    ax_bas.pcolormesh(cv1_axis, cv2_axis, basin_disp.T,
                      cmap=cmap_b, vmin=0.5, vmax=n_basins + 0.5, alpha=0.75)
    ax_bas.contour(cv1_axis, cv2_axis, disp_T, levels=15,
                   colors='k', linewidths=0.4, alpha=0.3)
    for m in minima:
        ax_bas.plot(m['cv1'], m['cv2'], '*', ms=12,
                    color='white', mec='black', mew=0.8, zorder=5)
        ax_bas.text(m['cv1'], m['cv2'], f"  {m['id']}",
                    fontsize=8, color='black', va='center', zorder=5)
    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap_b, norm=matplotlib.colors.Normalize(vmin=0.5, vmax=n_basins + 0.5)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_bas, ticks=range(1, n_basins + 1))
    cbar.set_label('Basin ID')
    ax_bas.set_xlabel(cv_names[0])
    ax_bas.set_ylabel(cv_names[1])
    title_tag = f' ({method_tag})' if method_tag else ''
    ax_bas.set_title(f'Basin assignments{title_tag}')

    plt.tight_layout()
    tag  = f'_{method_tag}' if method_tag else ''
    path = os.path.join(output_dir, f'fes_basins{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Analyze a PLUMED 2D FES: find local minima, define basins, '
            'and assign trajectory frames to each basin.  '
            'Periodic CVs (torsion angles) are handled automatically from '
            'the FES header.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--fes',           required=True,
                        help='PLUMED fes.dat file')
    parser.add_argument('--threshold',     required=True, type=float,
                        help='Energy threshold above global minimum (kJ/mol)')
    parser.add_argument('--sigma',         type=float, default=1.0,
                        help='Gaussian smoothing σ in grid units')
    parser.add_argument('--neighborhood',  type=int,   default=3,
                        help='Pixel neighborhood size for local-min detection')
    parser.add_argument('--colvar',        metavar='COLVAR',
                        help='PLUMED COLVAR file (CV values per frame)')
    parser.add_argument('--xtc',           metavar='TRAJ.XTC',
                        help='XTC trajectory file')
    parser.add_argument('--tpr',           metavar='TOPOL.TPR',
                        help='TPR topology (required when --xtc is given)')
    parser.add_argument('--basin-method',  choices=['watershed', 'contour', 'both'],
                        default='watershed')
    parser.add_argument('--contour-width', type=float, default=5.0,
                        help='Energy window above each minimum for contour method (kJ/mol)')
    parser.add_argument('--basin-xtc-window', type=float, default=None,
                        metavar='kJ/mol',
                        help='Only write XTC frames within this energy above the basin minimum '
                             '(default: include all frames in the basin)')
    parser.add_argument('--mep',           action='store_true',
                        help='Find minimum energy paths and saddle-point barriers between all basin pairs')
    parser.add_argument('--mep-all-pairs', action='store_true',
                        help='Plot MEPs for all basin pairs (default: adjacent pairs only)')
    parser.add_argument('--center-group',  default='protein', metavar='SEL',
                        help='MDAnalysis selection to center XTC output on (default: protein)')
    parser.add_argument('--no-center',     action='store_true',
                        help='Disable centering of XTC output')
    parser.add_argument('--cv1',           help='Override CV1 column name from COLVAR/FES')
    parser.add_argument('--cv2',           help='Override CV2 column name from COLVAR/FES')
    parser.add_argument('--output-dir',    default='fes_results')
    args = parser.parse_args()

    if args.xtc and not args.tpr:
        parser.error('--tpr is required when --xtc is given')
    if args.xtc and not args.colvar:
        parser.error('--colvar is required to map XTC frames to basins')

    os.makedirs(args.output_dir, exist_ok=True)

    # 1 ── Load & smooth ────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading FES: {args.fes}")
    cv1_axis, cv2_axis, fes_grid, cv_names, periodic = load_fes(args.fes)
    if args.cv1: cv_names[0] = args.cv1
    if args.cv2: cv_names[1] = args.cv2
    print(f"      Grid: {len(cv1_axis)} × {len(cv2_axis)}")
    print(f"      CVs:  {cv_names[0]} (periodic={periodic[0]})  ×  "
          f"{cv_names[1]} (periodic={periodic[1]})")

    print(f"[2/6] Smoothing  (σ = {args.sigma} grid units)")
    smoothed = smooth_fes(fes_grid, args.sigma, periodic)

    # 2 ── Find minima ──────────────────────────────────────────────────────────
    print(f"[3/6] Finding minima  (threshold = {args.threshold} kJ/mol above global min)")
    minima, global_min = find_minima(
        smoothed, cv1_axis, cv2_axis,
        args.threshold, args.neighborhood, periodic,
    )
    print(f"      Global minimum: {global_min:.3f} kJ/mol")
    print(f"      Found {len(minima)} basin(s):")
    for m in minima:
        print(f"        Basin {m['id']:2d}:  "
              f"{cv_names[0]} = {m['cv1']:9.4f}  "
              f"{cv_names[1]} = {m['cv2']:9.4f}  "
              f"ΔE = {m['delta_e']:.3f} kJ/mol")

    if not minima:
        sys.exit("No minima found. Try raising --threshold or reducing --sigma.")

    # 3 ── Define basins ────────────────────────────────────────────────────────
    print(f"[4/6] Defining basins  (method: {args.basin_method})")
    methods = ['watershed', 'contour'] if args.basin_method == 'both' else [args.basin_method]
    basins_map: dict[str, np.ndarray] = {}

    for method in methods:
        if method == 'watershed':
            basins_map[method] = define_basins_watershed(
                smoothed, minima, args.threshold, global_min, periodic
            )
        else:
            basins_map[method] = define_basins_contour(
                smoothed, minima, args.contour_width, periodic
            )
        np.savetxt(
            os.path.join(args.output_dir, f'basin_labels_{method}.dat'),
            basins_map[method], fmt='%d',
        )
        print(f"      Saved: basin_labels_{method}.dat")

    primary_basins = basins_map[methods[0]]

    # 4 ── Plot ────────────────────────────────────────────────────────────────
    print("[5/6] Plotting")
    multi = (args.basin_method == 'both')
    for method, basins in basins_map.items():
        plot_fes(
            smoothed, cv1_axis, cv2_axis, minima, basins,
            global_min, args.threshold, cv_names, args.output_dir,
            method_tag=method if multi else '',
        )

    # 5 ── Minimum energy paths ────────────────────────────────────────────────
    if args.mep:
        print("[MEP] Finding minimum energy paths between all basin pairs …")
        mep_results = find_mep(smoothed, minima, periodic)

        if args.mep_all_pairs:
            mep_to_plot = mep_results
        else:
            adj_pairs = find_adjacent_pairs(primary_basins, periodic)
            mep_to_plot = [r for r in mep_results
                           if (r['basin_a'], r['basin_b']) in adj_pairs]
            print(f"      {len(adj_pairs)} adjacent basin pairs found "
                  f"(use --mep-all-pairs to plot all {len(mep_results)})")

        plot_mep(smoothed, cv1_axis, cv2_axis, minima, mep_to_plot,
                 global_min, args.threshold, cv_names, args.output_dir)
        mep_df = write_mep_summary(mep_results, minima, cv1_axis, cv2_axis,
                                   cv_names, args.output_dir)
        print("\n      Barriers (sorted by forward barrier):")
        print(mep_df[['basin_a', 'basin_b',
                       'saddle_energy_kJ_mol',
                       'barrier_a_to_b_kJ_mol',
                       'barrier_b_to_a_kJ_mol']].to_string(index=False))
        print()

    # 6 ── Assign frames ────────────────────────────────────────────────────────
    if not args.colvar:
        print("[6/6] No --colvar provided — skipping frame assignment")
        print(f"\nDone. Outputs written to: {args.output_dir}/")
        return

    print("[6/6] Assigning trajectory frames")
    print(f"      Loading COLVAR: {args.colvar}")
    colvar_df = load_colvar(args.colvar, cv_names)
    print(f"      {len(colvar_df)} frames in COLVAR")

    # Basin assignment uses CV values — does not require XTC
    colvar_df = assign_frames_to_basins(
        colvar_df, primary_basins, cv1_axis, cv2_axis, cv_names,
        smoothed=smoothed,
    )
    n_assigned = (colvar_df['basin'] > 0).sum()
    print(f"      {n_assigned}/{len(colvar_df)} COLVAR frames assigned to a basin "
          f"({100*n_assigned/len(colvar_df):.1f}%)")

    # XTC matching: 1 XTC frame → 1 nearest COLVAR row (unique assignment)
    universe    = None
    xtc_basins  = None   # shape (n_xtc,), basin label per XTC frame
    center_ag   = None
    if args.xtc:
        universe, xtc_to_colvar, max_dt = match_xtc_to_colvar(
            colvar_df, args.xtc, args.tpr
        )
        xtc_basins = colvar_df['basin'].values[xtc_to_colvar]
        print(f"      Each XTC frame matched to its nearest COLVAR row "
              f"(max Δt = {max_dt:.3f} ps)")
        if not args.no_center:
            center_ag = universe.select_atoms(args.center_group)
            if len(center_ag) == 0:
                print(f"      WARNING: selection '{args.center_group}' matched 0 atoms — skipping centering")
                center_ag = None
            else:
                print(f"      Will center on '{args.center_group}' ({len(center_ag)} atoms) per frame")

    for m in minima:
        bdf       = colvar_df[colvar_df['basin'] == m['id']]
        list_path = write_frame_list(m, bdf, cv_names, args.output_dir)

        if universe is not None:
            xtc_frame_indices = np.where(xtc_basins == m['id'])[0]
            if args.basin_xtc_window is not None:
                energy_cutoff = m['energy'] + args.basin_xtc_window
                colvar_energies = colvar_df['fes_energy'].values
                xtc_frame_indices = xtc_frame_indices[
                    colvar_energies[xtc_to_colvar[xtc_frame_indices]] <= energy_cutoff
                ]
            xtc_out   = os.path.join(args.output_dir, f'basin_{m["id"]}.xtc')
            n_written = extract_xtc_frames(universe, xtc_frame_indices.tolist(), xtc_out,
                                           center_ag=center_ag)
            window_note = (f', ≤{args.basin_xtc_window:.1f} kJ/mol above min'
                           if args.basin_xtc_window is not None else '')
            print(f"      Basin {m['id']:2d}: {len(bdf):6d} COLVAR frames  "
                  f"({len(xtc_frame_indices)} XTC frames{window_note})  →  {xtc_out}")
        else:
            print(f"      Basin {m['id']:2d}: {len(bdf):6d} frames  →  {list_path}")

    summary_df = write_summary(minima, colvar_df, args.output_dir, cv_names)
    print("\n      Summary:")
    print(summary_df.to_string(index=False))

    colvar_df.to_csv(
        os.path.join(args.output_dir, 'colvar_assigned.csv'), index=False
    )

    print(f"\nDone. Outputs written to: {args.output_dir}/")


if __name__ == '__main__':
    main()
