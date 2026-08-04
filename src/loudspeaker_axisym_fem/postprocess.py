from __future__ import annotations
from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
from scipy import special

from .fem_solver import LoudspeakerParams, iso_1_24_frequencies, create_fem_model


def save_csv(path: str | Path, data: dict):
    path = Path(path)
    keys = [k for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 1]
    # Exclude nonmatching singleton arrays such as force_sign.
    n = len(data['f'])
    keys = [k for k in keys if len(data[k]) == n]
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = []
        for k in keys:
            if np.iscomplexobj(data[k]):
                header += [k + '_real', k + '_imag']
            else:
                header.append(k)
        w.writerow(header)
        for i in range(n):
            row = []
            for k in keys:
                v = data[k][i]
                if np.iscomplexobj(v):
                    row += [np.real(v), np.imag(v)]
                else:
                    row.append(float(v))
            w.writerow(row)


def plot_mesh(fem, outdir: Path, name='mesh_boundaries'):
    p = fem.params
    mesh = fem.mesh
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, linewidth=0.25)
    # boundary facet plotting
    labels = {101: 'outer PML boundary', 201: 'speaker front', 202: 'speaker back', 301: 'interior hard baffle', 302: 'closed back wall', 401: 'axis'}
    for tag, facets in sorted(fem.boundary_facets.items()):
        if tag not in labels or tag == 401:
            continue
        for fi in facets:
            nd = mesh.facets[:, fi]
            ax.plot(mesh.p[0, nd], mesh.p[1, nd], linewidth=2.0 if tag in [201,202,301,302] else 1.2)
    th = np.linspace(-np.pi/2, np.pi/2, 240)
    ax.plot(p.Rair*np.cos(th), p.Rair*np.sin(th), '--', linewidth=1.2, label='Rair/PML interface')
    ax.set_aspect('equal')
    ax.set_xlabel('r (m)')
    ax.set_ylabel('z (m)')
    ax.set_title('2D axisymmetric cracked FEM mesh and tagged boundaries')
    ax.set_xlim(-0.01, p.Rair + p.Rpml + 0.02)
    ax.set_ylim(-(p.Rair + p.Rpml + 0.02), p.Rair + p.Rpml + 0.02)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / f'{name}.png', dpi=180)
    plt.close(fig)


def _triangulation_physical(fem):
    mesh = fem.mesh
    tri = mesh.t.T
    centers = mesh.p[:, tri].mean(axis=2)
    rad = np.sqrt(centers[0]**2 + centers[1]**2)
    triang = mtri.Triangulation(mesh.p[0], mesh.p[1], tri)
    triang.set_mask(rad > fem.params.Rair + 1e-9)
    return triang


def plot_pressure_fields(fem, res: dict, outdir: Path, prefix='open'):
    triang = _triangulation_physical(fem)
    p = fem.params
    for f, field in res['fields'].items():
        fig, ax = plt.subplots(figsize=(6.6, 7.2))
        val = np.real(field)
        # symmetric color limits in physical domain
        tri = fem.mesh.t.T
        centers = fem.mesh.p[:, tri].mean(axis=2)
        mask_nodes = np.sqrt(fem.mesh.p[0]**2 + fem.mesh.p[1]**2) <= p.Rair + 1e-9
        vmax = np.nanpercentile(np.abs(val[mask_nodes]), 99.5)
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = np.max(np.abs(val)) + 1e-12
        tpc = ax.tripcolor(triang, val, shading='gouraud', vmin=-vmax, vmax=vmax)
        # overlay physical/PML interface and selected boundaries
        th = np.linspace(-np.pi/2, np.pi/2, 240)
        ax.plot(p.Rair*np.cos(th), p.Rair*np.sin(th), 'k--', linewidth=0.8)
        for tag in [201, 202, 301, 302]:
            if tag in fem.boundary_facets:
                for fi in fem.boundary_facets[tag]:
                    nd = fem.mesh.facets[:, fi]
                    ax.plot(fem.mesh.p[0, nd], fem.mesh.p[1, nd], 'k-', linewidth=0.8)
        ax.set_aspect('equal')
        ax.set_xlabel('r (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(f'Total acoustic pressure, FEM, {prefix}, f={f:g} Hz')
        ax.set_xlim(-0.005, p.Rair + 0.015)
        ax.set_ylim(-p.Rair - 0.015, p.Rair + 0.015)
        fig.colorbar(tpc, ax=ax, label='Pressure, peak real part (Pa)')
        fig.tight_layout()
        fig.savefig(outdir / f'field_pressure_{prefix}_{int(round(f))}Hz.png', dpi=180)
        plt.close(fig)


def plot_responses(freqs: np.ndarray, open_res: dict, closed_res: dict, params: LoudspeakerParams, outdir: Path):
    f = freqs
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0,0].semilogx(f, np.abs(open_res['u_D']), label='FEM open')
    axs[0,0].semilogx(f, np.abs(open_res['Qlf_D']/params.S_D), label='low-f approx')
    axs[0,0].semilogx(f, np.abs(open_res['Qhf_D']/params.S_D), label='high-f approx')
    axs[0,0].set(title='Diaphragm velocity', xlabel='f (Hz)', ylabel='Velocity amplitude (m/s)')
    axs[0,0].legend(fontsize=8); axs[0,0].grid(True, which='both', alpha=0.35)

    axs[0,1].semilogx(f, open_res['P_AR_front'], label='FEM front-side radiated power')
    axs[0,1].semilogx(f, open_res['P_AR_ana'], label='piston approximation')
    axs[0,1].set(title='Acoustic radiated power', xlabel='f (Hz)', ylabel='Power (W)')
    axs[0,1].legend(fontsize=8); axs[0,1].grid(True, which='both', alpha=0.35)

    axs[1,0].semilogx(f, open_res['P_E'], label='FEM open')
    axs[1,0].set(title='Electric input power', xlabel='f (Hz)', ylabel='Power (W)')
    axs[1,0].legend(fontsize=8); axs[1,0].grid(True, which='both', alpha=0.35)

    axs[1,1].semilogx(f, 100*open_res['eta'], label='FEM open')
    axs[1,1].semilogx(f, np.ones_like(f)*100*params.eta0, label='reference efficiency')
    axs[1,1].set(title='Driver efficiency', xlabel='f (Hz)', ylabel='Efficiency (%)')
    axs[1,1].legend(fontsize=8); axs[1,1].grid(True, which='both', alpha=0.35)
    fig.tight_layout(); fig.savefig(outdir/'response_velocity_power_efficiency.png', dpi=180); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0,0].semilogx(f, np.sqrt(0.5)*np.abs(open_res['Zac']*open_res['u_D']), label='FEM acoustic force RMS')
    axs[0,0].set(title='Pressure force on speaker cone', xlabel='f (Hz)', ylabel='Force RMS (N)')
    axs[0,0].legend(fontsize=8); axs[0,0].grid(True, which='both', alpha=0.35)

    axs[0,1].semilogx(f, np.abs(open_res['Zvoice']), label='|Z| open')
    axs[0,1].semilogx(f, np.real(open_res['Zvoice']), label='Re(Z) open')
    axs[0,1].semilogx(f, np.imag(open_res['Zvoice']), label='Im(Z) open')
    axs[0,1].semilogx(f, np.abs(closed_res['Zvoice']), '--', label='|Z| closed')
    axs[0,1].semilogx(f, np.real(closed_res['Zvoice']), '--', label='Re(Z) closed')
    axs[0,1].semilogx(f, np.imag(closed_res['Zvoice']), '--', label='Im(Z) closed')
    axs[0,1].set(title='Voice-coil impedance', xlabel='f (Hz)', ylabel='Impedance (ohm)')
    axs[0,1].legend(fontsize=7); axs[0,1].grid(True, which='both', alpha=0.35)

    axs[1,0].semilogx(f, open_res['SPL_1m'], label='FEM-coupled open')
    axs[1,0].semilogx(f, 20*np.log10(np.maximum(np.sqrt(2)*open_res['prms'],1e-300)/params.p_ref), label='piston approx')
    axs[1,0].semilogx(f, closed_res['SPL_1m'], label='FEM-coupled closed')
    axs[1,0].axvline(700, linewidth=1.0)
    axs[1,0].set(title='Sensitivity at 1 m', xlabel='f (Hz)', ylabel='Level (dB SPL rel. 1 V rms)')
    axs[1,0].legend(fontsize=8); axs[1,0].grid(True, which='both', alpha=0.35)

    axs[1,1].semilogx(f, open_res['phase_rel_plane'], label='phase rel. plane wave')
    axs[1,1].set(title='Phase relative to plane wave', xlabel='f (Hz)', ylabel='Phase (deg)')
    axs[1,1].legend(fontsize=8); axs[1,1].grid(True, which='both', alpha=0.35)
    fig.tight_layout(); fig.savefig(outdir/'response_force_impedance_sensitivity_phase.png', dpi=180); plt.close(fig)


def directivity_spl(freqs, u, angles_deg, p: LoudspeakerParams):
    freqs = np.asarray(freqs, dtype=float)
    angles = np.deg2rad(np.asarray(angles_deg, dtype=float))
    k = 2*np.pi*freqs[:,None]/p.c0
    x = k*p.a*np.sin(angles)[None,:]
    D = np.ones_like(x)
    mask = np.abs(x) > 1e-12
    D[mask] = 2*special.j1(x[mask])/x[mask]
    pressure = 1j*(2*np.pi*freqs[:,None])*p.rho0*u[:,None]*p.S_D*D/(2*np.pi*1.0)
    return 20*np.log10(np.maximum(np.abs(pressure)/np.sqrt(2),1e-300)/p.p_ref)


def plot_directivity(freqs, open_res, params, outdir: Path):
    angles = np.linspace(-90, 90, 181)
    spl = directivity_spl(freqs, open_res['u_D'], angles, params)
    spl_rel = spl - spl[:, [len(angles)//2]]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    pcm = ax.pcolormesh(angles, freqs, spl_rel, shading='auto')
    ax.set_yscale('log')
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Directivity, relative to on-axis, evaluated at 1 m')
    fig.colorbar(pcm, ax=ax, label='Relative level (dB)')
    fig.tight_layout(); fig.savefig(outdir/'directivity_relative.png', dpi=180); plt.close(fig)


def write_vtu(fem, fields: dict, outdir: Path, prefix='open'):
    pts = np.zeros((fem.N, 3))
    pts[:,0] = fem.mesh.p[0]
    pts[:,1] = fem.mesh.p[1]
    tri = fem.mesh.t.T
    for f, field in fields.items():
        point_data = {
            'pressure_real': np.real(field),
            'pressure_imag': np.imag(field),
            'pressure_abs': np.abs(field),
        }
        meshio.write_points_cells(outdir/f'field_{prefix}_{int(round(f))}Hz.vtu', pts, [('triangle', tri)], point_data=point_data)


def run_full(outdir: str | Path = 'outputs', h=0.014, h_speaker=0.0035):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    meshdir = outdir/'mesh'
    params = LoudspeakerParams()
    params.write_json(outdir/'parameters_resolved.json')
    freqs = iso_1_24_frequencies()
    open_fem = create_fem_model(meshdir, closed_back=False, params=params, h=h, h_speaker=h_speaker)
    closed_fem = create_fem_model(meshdir, closed_back=True, params=params, h=h, h_speaker=h_speaker)
    open_res = open_fem.solve_sweep(freqs, save_fields_at=(1000, 5000))
    closed_res = closed_fem.solve_sweep(freqs, save_fields_at=(1000,))
    save_csv(outdir/'frequency_response_open_fem.csv', open_res)
    save_csv(outdir/'frequency_response_closed_fem.csv', closed_res)
    plot_mesh(open_fem, outdir, name='mesh_boundaries_open')
    plot_mesh(closed_fem, outdir, name='mesh_boundaries_closed')
    plot_pressure_fields(open_fem, open_res, outdir, prefix='open')
    plot_pressure_fields(closed_fem, closed_res, outdir, prefix='closed')
    plot_responses(freqs, open_res, closed_res, params, outdir)
    plot_directivity(freqs, open_res, params, outdir)
    write_vtu(open_fem, open_res['fields'], outdir, prefix='open')
    write_vtu(closed_fem, closed_res['fields'], outdir, prefix='closed')

    idx_open = int(np.argmax(np.abs(open_res['Zvoice'])))
    idx_closed = int(np.argmax(np.abs(closed_res['Zvoice'])))
    checks = {
        'mesh_h_m': h,
        'open_nodes': int(open_fem.N),
        'open_triangles': int(open_fem.mesh.t.shape[1]),
        'closed_nodes': int(closed_fem.N),
        'closed_triangles': int(closed_fem.mesh.t.shape[1]),
        'Fs_parameters_Hz': float(params.Fs),
        'open_impedance_peak_frequency_Hz': float(freqs[idx_open]),
        'open_impedance_peak_abs_ohm': float(np.abs(open_res['Zvoice'][idx_open])),
        'closed_impedance_peak_frequency_Hz': float(freqs[idx_closed]),
        'closed_impedance_peak_abs_ohm': float(np.abs(closed_res['Zvoice'][idx_closed])),
        'eta0_percent': float(100*params.eta0),
        'open_SPL_700Hz_dB': float(np.interp(700.0, freqs, open_res['SPL_1m'])),
        'closed_SPL_700Hz_dB': float(np.interp(700.0, freqs, closed_res['SPL_1m'])),
        'force_sign_open': float(open_res['force_sign'][0]),
        'force_sign_closed': float(closed_res['force_sign'][0]),
    }
    (outdir/'sanity_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    return outdir
