"""Diagnostic analyzer for live 15-second Vision + IPC + Audio sessions.

Usage:
    python -m composer.diagnose artifacts/diag_session.jsonl [artifacts/diag_session.wav]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import wave
import numpy as np

from .recorder import KEY_LANDMARKS


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 4 or len(xs) != len(ys):
        return None
    ax = np.asarray(xs, dtype=np.float64)
    ay = np.asarray(ys, dtype=np.float64)
    sx = float(np.std(ax))
    sy = float(np.std(ay))
    if sx < 1e-6 or sy < 1e-6:
        return None
    return float(np.corrcoef(ax, ay)[0, 1])


def _read_wav_stereo(wav_path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(wav_path), 'rb') as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f'Solo se soporta PCM16 WAV (recibido {width * 8}-bit)')
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        return rate, data.reshape(-1, 2)
    return rate, np.column_stack([data, data])


def _db(linear: float) -> float:
    return 20.0 * math.log10(max(linear, 1e-7))


def analyze_session(jsonl_path: Path, wav_path: Path | None = None) -> dict:
    jsonl_path = Path(jsonl_path)
    header = {}
    footer = {}
    frames: list[dict] = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            kind = item.get('type')
            if kind == 'header':
                header = item
            elif kind == 'footer':
                footer = item
            elif kind == 'frame':
                frames.append(item)

    if wav_path is None and header.get('wav_path'):
        candidate = Path(header['wav_path'])
        if candidate.is_file():
            wav_path = candidate

    if not frames:
        raise ValueError(f'No hay frames registrados en {jsonl_path}')

    duration = max(0.001, frames[-1]['t'] - frames[0]['t'])
    new_pose_frames = [fr for fr in frames if fr.get('new_pose', True)]
    tracked_frames = [fr for fr in frames if fr['features'].get('tracked')]
    grace_frames = [fr for fr in frames if fr['features'].get('grace')]

    # 1) Landmark visibility statistics across camera frames.
    lm_stats = {}
    for _, name in KEY_LANDMARKS:
        vis_vals = [fr['landmarks'][name][3] for fr in new_pose_frames if fr.get('landmarks') and name in fr['landmarks']]
        if vis_vals:
            pct_ok = 100.0 * sum(v >= 0.5 for v in vis_vals) / len(vis_vals)
            lm_stats[name] = {'mean_vis': round(sum(vis_vals) / len(vis_vals), 3), 'pct_visible': round(pct_ok, 1)}
        else:
            lm_stats[name] = {'mean_vis': 0.0, 'pct_visible': 0.0}

    # 2) Feature ranges & discrete trigger counts.
    feat_keys = (
        'energy', 'left_height', 'right_height', 'expansion', 'fluidity',
        'shoulder_lift', 'shoulder_bounce', 'head_tilt', 'head_pitch',
        'torso_lean', 'torso_twist', 'hip_sway', 'body_level',
        'leg_energy', 'stance_width', 'body_bpm_confidence',
    )
    feat_ranges = {}
    for key in feat_keys:
        vals = [float(fr['features'].get(key, 0.0)) for fr in tracked_frames]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            feat_ranges[key] = {
                'min': round(float(np.min(arr)), 3),
                'mean': round(float(np.mean(arr)), 3),
                'p90': round(float(np.percentile(arr, 90)), 3),
                'max': round(float(np.max(arr)), 3),
            }
        else:
            feat_ranges[key] = {'min': 0.0, 'mean': 0.0, 'p90': 0.0, 'max': 0.0}

    triggers = {
        'left_strike_up': sum(1 for fr in frames if fr['features'].get('left_strike', 0) > 0),
        'left_strike_down': sum(1 for fr in frames if fr['features'].get('left_strike', 0) < 0),
        'right_strike_up': sum(1 for fr in frames if fr['features'].get('right_strike', 0) > 0),
        'right_strike_down': sum(1 for fr in frames if fr['features'].get('right_strike', 0) < 0),
        'left_leg_strikes': sum(1 for fr in frames if fr['features'].get('left_leg_strike')),
        'right_leg_strikes': sum(1 for fr in frames if fr['features'].get('right_leg_strike')),
        'pct_legs_visible': round(100.0 * sum(1 for fr in frames if fr['features'].get('legs_visible')) / len(frames), 1),
        'head_nods': sum(1 for fr in frames if fr['features'].get('head_nod')),
        'shoulder_drops': sum(1 for fr in frames if fr['features'].get('shoulder_drop')),
        'pct_head_harmony': round(100.0 * sum(1 for fr in frames if fr['composer'].get('head_harmony')) / len(frames), 1),
        'pct_shoulder_groove': round(100.0 * sum(1 for fr in frames if fr['composer'].get('shoulder_groove')) / len(frames), 1),
        'pct_bpm_synced': round(100.0 * sum(1 for fr in frames if fr['composer'].get('bpm_synced')) / len(frames), 1),
    }

    # 3) Musical event breakdown.
    limb_notes = {0: [], 1: [], 2: [], 3: []}
    drum_counts = {'kick_36': 0, 'snare_38': 0, 'hat_42': 0, 'open_hat_46': 0, 'other': 0}
    sections_seen: dict[str, int] = {}
    styles_seen: list[str] = []
    for fr in frames:
        sec = fr['composer'].get('section', 'UNKNOWN')
        sections_seen[sec] = sections_seen.get(sec, 0) + 1
        st = fr['composer'].get('style', '')
        if st and st not in styles_seen:
            styles_seen.append(st)
        for ev in fr.get('events', ()):
            if ev['kind'] == 0 and ev['limb'] in limb_notes:
                limb_notes[ev['limb']].append(ev['note'])
                if ev['limb'] == 3:
                    n = ev['note']
                    if n == 36:
                        drum_counts['kick_36'] += 1
                    elif n == 38:
                        drum_counts['snare_38'] += 1
                    elif n == 42:
                        drum_counts['hat_42'] += 1
                    elif n == 46:
                        drum_counts['open_hat_46'] += 1
                    else:
                        drum_counts['other'] += 1

    # 4) Audio WAV analysis & second-by-second timeline.
    audio_summary = None
    wav_sec_rms: dict[int, float] = {}
    if wav_path and Path(wav_path).is_file():
        rate, stereo = _read_wav_stereo(Path(wav_path))
        mono = 0.5 * (stereo[:, 0] + stereo[:, 1])
        rms_total = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
        peak_total = float(np.max(np.abs(stereo))) if len(stereo) else 0.0
        rms_l = float(np.sqrt(np.mean(stereo[:, 0] ** 2))) if len(stereo) else 0.0
        rms_r = float(np.sqrt(np.mean(stereo[:, 1] ** 2))) if len(stereo) else 0.0
        stereo_diff = float(np.sqrt(np.mean((stereo[:, 0] - stereo[:, 1]) ** 2))) if len(stereo) else 0.0

        # Spectral energy distribution (Sub <100, Low 100-250, Mid 250-2000, High >2000 Hz).
        spec = np.abs(np.fft.rfft(mono)) ** 2
        freqs = np.fft.rfftfreq(len(mono), 1.0 / rate)
        total_spec = float(np.sum(spec)) + 1e-12
        bands = {
            'sub_lt_100hz_pct': round(100.0 * float(np.sum(spec[freqs < 100])) / total_spec, 1),
            'low_100_250hz_pct': round(100.0 * float(np.sum(spec[(freqs >= 100) & (freqs < 250)])) / total_spec, 1),
            'mid_250_2k_pct': round(100.0 * float(np.sum(spec[(freqs >= 250) & (freqs < 2000)])) / total_spec, 1),
            'high_gt_2k_pct': round(100.0 * float(np.sum(spec[freqs >= 2000])) / total_spec, 1),
        }

        n_secs = max(1, int(math.ceil(len(mono) / rate)))
        silent_secs = 0
        for s in range(n_secs):
            seg = mono[s * rate:(s + 1) * rate]
            r = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
            wav_sec_rms[s] = r
            if _db(r) < -45.0:
                silent_secs += 1

        audio_summary = {
            'wav_path': str(wav_path),
            'duration_s': round(len(mono) / rate, 2),
            'rms_dbfs': round(_db(rms_total), 2),
            'peak_dbfs': round(_db(peak_total), 2),
            'clipping_samples': int(np.sum(np.abs(stereo) >= 0.999)),
            'stereo_width_ratio': round(stereo_diff / max(rms_total, 1e-6), 3),
            'lr_balance_db': round(_db(rms_r) - _db(rms_l), 2),
            'silent_seconds': silent_secs,
            'spectral_bands': bands,
        }

    # Build 1-second timeline and compute Vision <-> Audio correlations.
    max_sec = int(math.ceil(duration))
    timeline = []
    corr_energy_x, corr_rms_y = [], []
    corr_rh_x, corr_pitch_y = [], []
    corr_lh_x, corr_cut_y = [], []
    for s in range(max_sec):
        bucket = [fr for fr in frames if s <= fr['t'] < s + 1]
        if not bucket:
            continue
        avg_e = sum(fr['features']['energy'] for fr in bucket) / len(bucket)
        avg_rh = sum(fr['features']['right_height'] for fr in bucket) / len(bucket)
        avg_lh = sum(fr['features']['left_height'] for fr in bucket) / len(bucket)
        max_sh = max(fr['features'].get('shoulder_lift', 0.0) for fr in bucket)
        max_ht = max(abs(fr['features'].get('head_tilt', 0.0)) for fr in bucket)
        avg_cut = sum(fr['controls']['cutoff'][0] for fr in bucket) / len(bucket)
        sec_lead = [ev['note'] for fr in bucket for ev in fr.get('events', ()) if ev['kind'] == 0 and ev['limb'] == 1]
        sec_notes = sum(1 for fr in bucket for ev in fr.get('events', ()) if ev['kind'] == 0)
        dom_sec = max({fr['composer']['section'] for fr in bucket}, key=lambda k: sum(fr['composer']['section'] == k for fr in bucket))
        avg_bpm = sum(fr['composer']['bpm'] for fr in bucket) / len(bucket)

        audio_db = round(_db(wav_sec_rms[s]), 1) if s in wav_sec_rms else None
        if s in wav_sec_rms:
            corr_energy_x.append(avg_e)
            corr_rms_y.append(wav_sec_rms[s])
        if sec_lead:
            corr_rh_x.append(avg_rh)
            corr_pitch_y.append(sum(sec_lead) / len(sec_lead))
        corr_lh_x.append(avg_lh)
        corr_cut_y.append(avg_cut)

        timeline.append({
            'sec': s,
            'energy': round(avg_e, 2),
            'right_h': round(avg_rh, 2),
            'left_h': round(avg_lh, 2),
            'shrug': round(max_sh, 2),
            'head_tilt': round(max_ht, 2),
            'section': dom_sec,
            'bpm': round(avg_bpm, 1),
            'notes': sec_notes,
            'lead_pitch': round(sum(sec_lead) / len(sec_lead), 1) if sec_lead else None,
            'audio_dbfs': audio_db,
        })

    correlations = {
        'energy_vs_audio_rms': _pearson(corr_energy_x, corr_rms_y),
        'right_height_vs_lead_pitch': _pearson(corr_rh_x, corr_pitch_y),
        'left_height_vs_pad_cutoff': _pearson(corr_lh_x, corr_cut_y),
    }

    # 5) Automatic issue detection (actionable diagnostics!).
    findings: list[str] = []
    pose_fps = len(new_pose_frames) / duration
    if pose_fps < 18.0:
        findings.append(f'FPS de pose bajos ({pose_fps:.1f} FPS): prueba `--pose-model lite` o mejor iluminación para capturar brazos rápidos.')
    pct_tracked = 100.0 * len(tracked_frames) / len(frames)
    if pct_tracked < 85.0:
        findings.append(f'Tracking intermitente ({pct_tracked:.1f}% de frames válidos): revisa encuadre de hombros, caderas y muñecas.')
    if lm_stats['l_ear']['pct_visible'] < 50.0 or lm_stats['r_ear']['pct_visible'] < 50.0:
        findings.append(f'Orejas poco visibles (izq {lm_stats["l_ear"]["pct_visible"]}%, der {lm_stats["r_ear"]["pct_visible"]}%): `head_tilt` cae al fallback de nariz y puede perder precisión.')
    if 0.15 < feat_ranges['shoulder_lift']['max'] < 0.48:
        findings.append(f'`shoulder_lift` llegó a {feat_ranges["shoulder_lift"]["max"]:.2f} pero el umbral de BUILDUP es 0.48 (podría bajarse el umbral para tu postura).')
    if 0.12 < max(abs(feat_ranges['head_tilt']['min']), abs(feat_ranges['head_tilt']['max'])) < 0.28:
        findings.append('`head_tilt` se quedó debajo del umbral 0.28 para `HEAD HARMONY` (conviene aumentar sensibilidad).')
    if triggers['pct_legs_visible'] < 20.0:
        findings.append(
            f'Piernas poco visibles en cámara ({triggers["pct_legs_visible"]}%): aléjate un poco o inclina la cámara para incluir rodillas/tobillos y activar kick/snare con los pies.'
        )
    if triggers['pct_bpm_synced'] == 0.0 and feat_ranges['energy']['max'] > 0.35:
        findings.append('El BPM nunca enganchó con el cuerpo (`pct_bpm_synced = 0%`): revisa `body_bpm_confidence` máx '
                        f'({feat_ranges["body_bpm_confidence"]["max"]:.2f} vs umbral 0.48).')
    if audio_summary:
        if audio_summary['rms_dbfs'] < -28.0:
            findings.append(f'Nivel de audio muy bajo ({audio_summary["rms_dbfs"]} dBFS): conviene subir ganancia master o dinámica de voces.')
        if audio_summary['clipping_samples'] > 20:
            findings.append(f'Se detectaron {audio_summary["clipping_samples"]} muestras en clipping (≥0.999).')
        if audio_summary['spectral_bands']['high_gt_2k_pct'] < 3.0:
            findings.append(f'Audio muy opaco/oscuro (agudos >2 kHz = {audio_summary["spectral_bands"]["high_gt_2k_pct"]}%): el filtro pasa-bajos está demasiado cerrado.')
    if not findings:
        findings.append('Sin anomalías críticas detectadas automáticamente; revisa la tabla segundo-a-segundo y el WAV.')

    return {
        'jsonl_path': str(jsonl_path),
        'duration_s': round(duration, 2),
        'control_frames': len(frames),
        'pose_frames': len(new_pose_frames),
        'pose_fps': round(pose_fps, 1),
        'pct_tracked': round(pct_tracked, 1),
        'pct_grace': round(100.0 * len(grace_frames) / len(frames), 1),
        'styles_seen': styles_seen,
        'sections_seen': sections_seen,
        'landmark_visibility': lm_stats,
        'feature_ranges': feat_ranges,
        'triggers': triggers,
        'note_counts': {
            'chords_limb0': len(limb_notes[0]),
            'melody_limb1': len(limb_notes[1]),
            'bass_limb2': len(limb_notes[2]),
            'drums_limb3': len(limb_notes[3]),
            'drums_detail': drum_counts,
            'melody_midi_range': [min(limb_notes[1]), max(limb_notes[1])] if limb_notes[1] else None,
        },
        'audio': audio_summary,
        'correlations': {k: (round(v, 3) if v is not None else None) for k, v in correlations.items()},
        'timeline': timeline,
        'findings': findings,
        'engine_report': footer.get('report', {}),
    }


def format_report_markdown(diag: dict) -> str:
    lines = [
        '# Reporte de Diagnóstico en Vivo — CV-DAW',
        '',
        f'- **Duración**: `{diag["duration_s"]} s` | **Pose FPS**: `{diag["pose_fps"]}` | **Tracking**: `{diag["pct_tracked"]}%` (Grace: `{diag["pct_grace"]}%`)',
        f'- **Estilos**: `{", ".join(diag["styles_seen"])}` | **Secciones**: `{diag["sections_seen"]}`',
        '',
        '## Hallazgos Automáticos',
    ]
    for item in diag['findings']:
        lines.append(f'- {item}')

    if diag['audio']:
        a = diag['audio']
        b = a['spectral_bands']
        lines += [
            '',
            '## Análisis Acústico del WAV',
            f'- **RMS**: `{a["rms_dbfs"]} dBFS` | **Pico**: `{a["peak_dbfs"]} dBFS` | **Clipping**: `{a["clipping_samples"]}` muestras | **Segundos silenciosos**: `{a["silent_seconds"]}`',
            f'- **Anchura estéreo**: `{a["stereo_width_ratio"]}` | **Balance L/R**: `{a["lr_balance_db"]:+.2f} dB`',
            f'- **Espectro**: Sub `<100Hz` **{b["sub_lt_100hz_pct"]}%** · Graves `100-250Hz` **{b["low_100_250hz_pct"]}%** · Medios `250-2kHz` **{b["mid_250_2k_pct"]}%** · Agudos `>2kHz` **{b["high_gt_2k_pct"]}%**',
        ]

    c = diag['correlations']
    lines += [
        '',
        '## Correlaciones Visión ↔ Música',
        f'- `energy ↔ audio RMS`: `{c["energy_vs_audio_rms"]}`',
        f'- `right_height ↔ pitch melodía`: `{c["right_height_vs_lead_pitch"]}`',
        f'- `left_height ↔ pad cutoff`: `{c["left_height_vs_pad_cutoff"]}`',
        '',
        '## Disparos y Notas Enviadas al Motor C++',
        f'- **Triggers**: `{diag["triggers"]}`',
        f'- **Notas por bus**: `{diag["note_counts"]}`',
        '',
        '## Visibilidad de Landmarks Clave',
        '| Punto | % Visible (≥0.5) | Visibilidad Media |',
        '| :--- | :---: | :---: |',
    ]
    for name, st in diag['landmark_visibility'].items():
        lines.append(f'| `{name}` | {st["pct_visible"]}% | {st["mean_vis"]} |')

    lines += [
        '',
        '## Rangos de Rasgos Corporales',
        '| Rasgo | Min | Media | P90 | Max |',
        '| :--- | :---: | :---: | :---: | :---: |',
    ]
    for key, r in diag['feature_ranges'].items():
        lines.append(f'| `{key}` | {r["min"]} | {r["mean"]} | {r["p90"]} | {r["max"]} |')

    lines += [
        '',
        '## Línea de Tiempo (Segundo a Segundo)',
        '| Seg | Energía | Brazo Der | Brazo Izq | Shrug | Head Tilt | Sección | BPM | Notas | Pitch Melodía | Audio (dBFS) |',
        '| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |',
    ]
    for row in diag['timeline']:
        lines.append(
            f'| {row["sec"]} | {row["energy"]} | {row["right_h"]} | {row["left_h"]} | {row["shrug"]} | '
            f'{row["head_tilt"]} | {row["section"]} | {row["bpm"]} | {row["notes"]} | '
            f'{row["lead_pitch"] if row["lead_pitch"] is not None else "-"} | '
            f'{row["audio_dbfs"] if row["audio_dbfs"] is not None else "-"} |'
        )
    return '\n'.join(lines) + '\n'


def write_report(jsonl_path: Path, wav_path: Path | None = None, out_md: Path | None = None) -> tuple[dict, Path]:
    diag = analyze_session(jsonl_path, wav_path)
    report_path = out_md or jsonl_path.with_name(jsonl_path.stem + '_report.md')
    report_path.write_text(format_report_markdown(diag), encoding='utf-8')
    return diag, report_path


def main():
    p = argparse.ArgumentParser(description='Analiza una sesión grabada (.jsonl + .wav) de CV-DAW')
    p.add_argument('jsonl', type=Path, help='Archivo .jsonl generado con --diagnose')
    p.add_argument('wav', nargs='?', type=Path, default=None, help='Archivo .wav correspondiente (opcional si está en header)')
    p.add_argument('--out', type=Path, default=None, help='Ruta de salida para el reporte .md')
    args = p.parse_args()
    diag, report_path = write_report(args.jsonl, args.wav, args.out)
    print(f'Reporte guardado en: {report_path}')
    for finding in diag['findings']:
        print(f'  * {finding}')


if __name__ == '__main__':
    main()
