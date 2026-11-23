"""Studio performance HUD and live keyboard controls for CV-DAW (Pose-Only Full-Body Moves)."""
from __future__ import annotations

import time
from ..styles import STYLE_ORDER


def handle_window_key(key: int, composer) -> bool:
    """Handle live keyboard shortcuts for styles, beat toggle, BPM lock, and exit. Returns True to quit."""
    if key == ord('q'):
        return True
    direct_keys = (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'))
    if key in direct_keys:
        idx = direct_keys.index(key)
        if idx < len(STYLE_ORDER):
            composer.set_genre(STYLE_ORDER[idx])
            print(f'Estilo [{idx + 1}/{len(STYLE_ORDER)}]: {composer.preset} ({composer.bpm:.0f} BPM)', flush=True)
    elif key in (ord('n'), ord('N'), ord(']')):
        composer.cycle_genre(1)
        idx = STYLE_ORDER.index(composer.genre_key) + 1 if composer.genre_key in STYLE_ORDER else 1
        print(f'Estilo [{idx}/{len(STYLE_ORDER)}]: {composer.preset} ({composer.bpm:.0f} BPM)', flush=True)
    elif key in (ord('p'), ord('P'), ord('[')):
        composer.cycle_genre(-1)
        idx = STYLE_ORDER.index(composer.genre_key) + 1 if composer.genre_key in STYLE_ORDER else 1
        print(f'Estilo [{idx}/{len(STYLE_ORDER)}]: {composer.preset} ({composer.bpm:.0f} BPM)', flush=True)
    elif key in (ord('t'), ord('T')):
        locked = composer.toggle_tempo_lock()
        print(f'BPM {"bloqueado" if locked else "sincronizado con el cuerpo"} ({composer.bpm:.0f} BPM)', flush=True)
    elif key in (ord('b'), ord('B'), ord(' ')):
        active = composer.toggle_beat()
        print(f'Beat: {"ON" if active else "OFF"}', flush=True)
    return False


def draw_hud(cv2, image, features, composer, flash_state=None, rec_status=None):
    """Draw a clean, translucent studio HUD highlighting recognized full-body & leg moves."""
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 142), (14, 16, 22), -1)
    cv2.rectangle(overlay, (0, h - 30), (w, h), (14, 16, 22), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)

    if not features.tracked:
        status_text, status_color = 'BUSCANDO POSE', (70, 90, 240)
    elif features.grace:
        status_text, status_color = 'GRACE', (40, 195, 255)
    else:
        status_text, status_color = 'BODY LIVE', (90, 235, 135)

    cv2.circle(image, (20, 22), 7, status_color, -1, cv2.LINE_AA)
    bpm_tag = 'LOCK' if getattr(composer, 'tempo_lock', False) else (
        f'SYNC:{getattr(composer, "bpm_sync_source", "").upper()}'
        if getattr(composer, 'bpm_synced', False)
        else 'AUTO'
    )
    style_idx = (STYLE_ORDER.index(composer.genre_key) + 1) if getattr(composer, 'genre_key', '') in STYLE_ORDER else 1
    section = getattr(composer, 'section', 'INTRO')
    bar_num = getattr(composer, 'bar_in_cycle', 0) + 1
    chord_name = getattr(composer, 'current_chord_name', '')
    beat_tag = 'X-CUT' if features.arms_crossed else (
        'FREEZE' if features.body_frozen else ('BEAT:ON' if getattr(composer, 'beat_active', True) else 'BEAT:OFF')
    )
    title = (
        f'[{style_idx}/{len(STYLE_ORDER)}] {getattr(composer, "preset", "CV-DAW")} | '
        f'{composer.bpm:.0f} BPM ({bpm_tag}) | {chord_name} | {section} [{bar_num}/4] | {beat_tag}'
    )
    cv2.putText(image, title, (34, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 244, 248), 2, cv2.LINE_AA)
    if rec_status:
        cv2.circle(image, (w - 245, 22), 6, (40, 40, 250), -1, cv2.LINE_AA)
        cv2.putText(image, rec_status, (w - 232, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 90, 255), 2, cv2.LINE_AA)
    cv2.putText(image, status_text, (w - 118, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

    footer = '[1-6] Estilo  [Space] Beat  [T] BPM  |  Punch · Brazos Arriba=Wave · X=Cut · ArmWave · Rodilla/Sentadilla/Salto  [q] Salir'
    cv2.putText(image, footer, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (210, 218, 230), 1, cv2.LINE_AA)

    if not features.tracked:
        cv2.putText(
            image,
            'Necesito ver hombros, caderas y brazos (alejate un poco para incluir rodillas/piernas)',
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 190, 210),
            1,
            cv2.LINE_AA,
        )
        return image

    legs_label = f'Piernas ({features.legs_source[:3].upper()})' if features.legs_visible else 'Piernas (NO)'
    meters = (
        ('Energia', features.energy, (60, 210, 255)),
        ('Soundwave', features.soundwave_rise, (235, 130, 255)),
        ('Squat Dive', features.squat_depth, (80, 220, 255)),
        ('Espacio', features.expansion, (90, 230, 175)),
        (legs_label, features.leg_energy if features.legs_visible else 0.0, (255, 185, 70)),
    )
    bar_w = max(65, (w - 48) // len(meters) - 14)
    for idx, (label, val, color) in enumerate(meters):
        x0 = 16 + idx * (bar_w + 14)
        y0 = 56
        cv2.putText(image, f'{label} {val:.2f}', (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (210, 216, 224), 1, cv2.LINE_AA)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + 10), (55, 60, 72), -1)
        fill = int(bar_w * max(0.0, min(1.0, val)))
        if fill > 0:
            cv2.rectangle(image, (x0, y0), (x0 + fill, y0 + 10), color, -1)

    body_line = (
        f'Filtro Izq:{features.left_height:.2f}  Lead Der:{features.right_height:.2f}  |  '
        f'Cabeza:{features.head_tilt:+.2f}  |  Cadera:{features.hip_sway:+.2f}  |  '
        f'Stance:{features.stance_width:.2f}'
    )
    cv2.putText(image, body_line, (16, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (185, 205, 225), 1, cv2.LINE_AA)

    badges = []
    if features.active_move_label:
        badges.append((features.active_move_label, (95, 255, 175)))
    if getattr(composer, 'head_harmony_active', False):
        badges.append(('HARMONY + ECHO', (235, 145, 255)))
    if flash_state:
        now = time.monotonic()
        txt, until = flash_state.get('move', ('', 0.0))
        if now < until and txt and txt != features.active_move_label:
            badges.append((txt, (80, 230, 255)))

    bx = 16
    for text, color in badges:
        cv2.putText(image, text, (bx, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        bx += max(140, len(text) * 11 + 18)
    return image
