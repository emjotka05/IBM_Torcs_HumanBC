import pygame
import snakeoil3_gym as snakeoil3
import json
import os
import signal
import time
import numpy as np
from train import get_state, _apply_gears, DATA_FILE, LEGACY_STATE_DIM, PORT, STATE_DIM


class KeyboardInput:
    def __init__(self):
        self._user32 = None
        if os.name == "nt":
            try:
                import ctypes
                self._user32 = ctypes.windll.user32
            except Exception:
                self._user32 = None

    @property
    def uses_global_keys(self) -> bool:
        return self._user32 is not None

    def pressed(self, *keys) -> bool:
        if self._user32 is not None:
            return any(self._is_windows_key_down(key) for key in keys)

        pygame_keys = pygame.key.get_pressed()
        return any(pygame_keys[key] for key in keys)

    def _is_windows_key_down(self, key) -> bool:
        vk = {
            pygame.K_LEFT: 0x25,
            pygame.K_UP: 0x26,
            pygame.K_RIGHT: 0x27,
            pygame.K_DOWN: 0x28,
            pygame.K_a: 0x41,
            pygame.K_d: 0x44,
            pygame.K_s: 0x53,
            pygame.K_w: 0x57,
            pygame.K_ESCAPE: 0x1B,
            pygame.K_F8: 0x77,
            pygame.K_F9: 0x78,
            pygame.K_BACKSPACE: 0x08,
        }.get(key)
        return bool(vk and (self._user32.GetAsyncKeyState(vk) & 0x8000))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SMOOTH CAR CONTROLLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SmoothCarController:
    """
    Kontroler pojazdu oparty na trzech systemach fizycznych:

      1. Kierownica — model sprężyna-tłumik z bezwładnością
         Kierownica NIE skacze natychmiast do max. Posiada prędkość kątową
         (steer_vel), która rośnie powoli pod wpływem klawisza, jest tłumiona
         każdą klatką i ściągana sprężyną do centrum po puszczeniu klawisza.
         Szybkość reakcji maleje wraz z prędkością (auto staje się 'stabilniejsze').

      2. Gaz — progresywny ramp z launch control
         Bardzo wolne narastanie przy ruszaniu (v < 20 km/h) zapobiega spinaniu
         kół. Przy normalnej jeździe ramp jest szybszy, ale wciąż płynny (~2 sek
         do pełnego gazu). Jednoczesne gaz+hamulec jest zablokowane.

      3. Traction Circle + Stability Control
         Dostępna przyczepność jest WSPÓLNA dla skrętu i gazu:
           lat_cost  = |steer|^EXP × SCALE   (koszt boczny, nieliniowy)
           long_budget = 1 − lat_cost          (co zostaje na przyspieszenie)
         Stability Control wykrywa ślizg przez gwałtowną zmianę kąta 'angle'
         (kąt osi pojazdu vs oś toru). Przy ślizgu SC ucina gaz płynnie.
    """

    # ── Kierownica ───────────────────────────────────────────────────────────
    STEER_INPUT_RATE   = 0.030   # Szybkość reakcji na klawisz [/klatkę]
    STEER_DAMPING      = 0.68    # Tłumik (0 = brak bezwładności, 1 = brak tłumienia)
    STEER_SPRING       = 0.10    # Siła sprężyny powrotu do centrum
    STEER_SPEED_REF    = 85.0    # [km/h] — powyżej tej prędkości rate spada 2×
    STEER_OUTPUT_GAMMA = 0.80    # Krzywa wyjściowa: < 1.0 = precyzja w centrum

    # ── Gaz ──────────────────────────────────────────────────────────────────
    ACCEL_RATE_LAUNCH  = 0.012   # Narastanie gazu przy ruszaniu (v < 20 km/h)
    ACCEL_RATE_CRUISE  = 0.028   # Narastanie gazu przy normalnej jeździe
    ACCEL_RELEASE      = 0.10    # Szybkość odpuszczenia gazu

    # ── Hamulec ──────────────────────────────────────────────────────────────
    BRAKE_RATE         = 0.065   # Narastanie hamulca
    BRAKE_RELEASE      = 0.22    # Odpuszczenie hamulca
    BRAKE_MIN_EFF      = 0.55    # ABS: minimalna efektywność przy pełnym skręcie

    # ── Traction Circle ───────────────────────────────────────────────────────
    LAT_GRIP_EXP       = 0.55    # Wykładnik kosztu bocznego (< 1 = małe skręty tanie)
    LAT_GRIP_SCALE     = 0.72    # Max koszt pełnego skrętu (reszta = budżet na gaz)

    # ── Stability Control ────────────────────────────────────────────────────
    SLIP_THRESHOLD     = 0.045   # [rad/klatkę] próg detekcji ślizgu bocznego
    SLIP_RISE          = 0.25    # Szybkość wzrostu wskaźnika ślizgu
    SLIP_FALL          = 0.04    # Szybkość zaniku wskaźnika (histereza — SC nie odpada gwałtownie)
    SLIP_TC_GAIN       = 1.30    # Agresywność cięcia gazu przez SC

    def __init__(self):
        self.steer        = 0.0
        self.steer_vel    = 0.0
        self.accel        = 0.0
        self.brake        = 0.0
        self._prev_angle  = 0.0
        self.slip_level   = 0.0   # 0 = brak ślizgu, 1 = pełny ślizg

    def update(self, keys, speed_x: float, angle: float):
        """
        Przetwarza klawisze + stan sensorów → wyjścia gotowe do TORCS.

        Args:
            keys:    pygame.key.get_pressed()
            speed_x: prędkość wzdłużna [km/h] z TORCS S['speedX']
            angle:   kąt osi pojazdu vs oś toru [rad] z TORCS S['angle']

        Returns:
            (final_steer, final_accel, final_brake)
        """
        spd = max(0.0, float(speed_x))
        ang = float(angle)

        # ── 1. KIEROWNICA: system sprężyna-tłumik ────────────────────────────
        #
        #  Szybkość reakcji spada liniowo z prędkością:
        #    speed_factor = 1 / (1 + v / STEER_SPEED_REF)
        #  Przy v=0:    factor=1.0  (pełna czułość)
        #  Przy v=85:   factor=0.5  (połowa czułości)
        #  Przy v=170:  factor=0.33 (1/3 czułości)
        #
        speed_factor = 1.0 / (1.0 + spd / self.STEER_SPEED_REF)
        input_rate   = self.STEER_INPUT_RATE * speed_factor

        left  = keys.pressed(pygame.K_LEFT, pygame.K_a)
        right = keys.pressed(pygame.K_RIGHT, pygame.K_d)

        if left and not right:
            # Akceleracja kątowa kierownicy w lewo
            self.steer_vel += input_rate
        elif right and not left:
            # Akceleracja kątowa kierownicy w prawo
            self.steer_vel -= input_rate
        else:
            # Sprężyna: siła proporcjonalna do odchylenia od centrum
            # Im dalej kierownica od 0, tym silniej jest ściągana z powrotem
            self.steer_vel -= self.steer * self.STEER_SPRING

        # Tłumik bezwładności (aplikowany po sile wejściowej)
        self.steer_vel *= self.STEER_DAMPING
        # Klamp max prędkości kątowej (sprężyna nie może rozkręcić oscylacji)
        self.steer_vel  = np.clip(self.steer_vel, -input_rate * 2.5, input_rate * 2.5)
        # Integracja: prędkość kątowa → pozycja kierownicy
        self.steer = np.clip(self.steer + self.steer_vel, -1.0, 1.0)

        # ── 2. PEDAŁY: progresywne narastanie z launch control ───────────────
        #
        #  Launch Control aktywny gdy v < 20 km/h:
        #    ACCEL_RATE_LAUNCH = 0.012  → pełny gaz w ~83 klatkach ≈ 1.7 sek
        #  Normalna jazda:
        #    ACCEL_RATE_CRUISE = 0.028  → pełny gaz w ~36 klatkach ≈ 0.7 sek
        #
        accel_rate = self.ACCEL_RATE_LAUNCH if spd < 20.0 else self.ACCEL_RATE_CRUISE

        if keys.pressed(pygame.K_UP, pygame.K_w):
            self.accel = min(1.0, self.accel + accel_rate)
        else:
            self.accel = max(0.0, self.accel - self.ACCEL_RELEASE)

        if keys.pressed(pygame.K_DOWN, pygame.K_s):
            self.brake = min(1.0, self.brake + self.BRAKE_RATE)
        else:
            self.brake = max(0.0, self.brake - self.BRAKE_RELEASE)

        # Blokada gaz+hamulec (jak w prawdziwym samochodzie bez launch control)
        if self.brake > 0.05:
            self.accel = max(0.0, self.accel - self.brake * 0.8)

        # ── 3. STABILITY CONTROL: detekcja ślizgu przez zmianę angle ─────────
        #
        #  'angle' to kąt osi pojazdu względem osi toru w radianach.
        #  Przy normalnej jeździe zmienia się powoli.
        #  Przy ślizgu bocznym lub korkociągu — skacze gwałtownie.
        #  Filtrujemy to histerezą: SLIP_RISE >> SLIP_FALL,
        #  dzięki czemu SC włącza się szybko, ale wyłącza płynnie.
        #
        angle_delta = abs(ang - self._prev_angle)
        self._prev_angle = ang

        if angle_delta > self.SLIP_THRESHOLD:
            self.slip_level = min(1.0, self.slip_level + self.SLIP_RISE)
        else:
            self.slip_level = max(0.0, self.slip_level - self.SLIP_FALL)

        # ── 4. TRACTION CIRCLE: podział dostępnej przyczepności ───────────────
        #
        #  Model:  lat_cost  = |steer|^LAT_GRIP_EXP × LAT_GRIP_SCALE
        #          long_budget = 1 − lat_cost
        #
        #  Wykładnik < 1 (tu 0.55) sprawia, że małe skręty są 'tanie':
        #    steer=0.1 → lat_cost=0.14  → 86% budżetu na gaz
        #    steer=0.5 → lat_cost=0.50  → 50% budżetu na gaz
        #    steer=1.0 → lat_cost=0.72  → 28% budżetu na gaz
        #
        lat_cost    = (abs(self.steer) ** self.LAT_GRIP_EXP) * self.LAT_GRIP_SCALE
        long_budget = max(0.0, 1.0 - lat_cost)

        # Redukcja gazu: traction circle × stability control
        sc_cut      = min(1.0, self.slip_level * self.SLIP_TC_GAIN)
        final_accel = np.clip(self.accel * long_budget * (1.0 - sc_cut), 0.0, 1.0)

        # ABS: efektywność hamowania maleje przy skręcie (koła muszą też skręcać)
        brake_eff   = max(self.BRAKE_MIN_EFF,
                          1.0 - abs(self.steer) * (1.0 - self.BRAKE_MIN_EFF))
        final_brake = self.brake * brake_eff

        # ── 5. PROGRESYWNA KRZYWA WYJŚCIOWA KIEROWNICY ───────────────────────
        #
        #  |steer|^0.80 daje 'przyspieszenie' w centrum zakresu:
        #    Wewnętrzna pozycja 0.3 → wyjście 0.35  (centrum, precyzja)
        #    Wewnętrzna pozycja 0.8 → wyjście 0.83  (krawędź, prawie liniowo)
        #
        sgn         = np.sign(self.steer)
        final_steer = float(sgn * (abs(self.steer) ** self.STEER_OUTPUT_GAMMA))

        return final_steer, float(final_accel), float(final_brake)

    @property
    def is_sc_active(self) -> bool:
        """True jeśli Stability Control aktywnie interweniuje."""
        return self.slip_level > 0.15

    @property
    def is_lc_active(self) -> bool:
        """True jeśli aktywny Launch Control (wolny ramp przy ruszaniu)."""
        return self._prev_speed_x < 20.0 if hasattr(self, '_prev_speed_x') else False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UI RENDERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bar(surface, x, y, w, h, value, color, bg=(50, 52, 58)):
    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=3)
    if value > 0.001:
        pygame.draw.rect(surface, color, (x, y, int(w * value), h), border_radius=3)


def draw_ui(screen, font, small, ctrl: SmoothCarController,
            fs, fa, fb, speed_x, dist, samples, lap, speed_x_raw):
    W, H = screen.get_size()
    screen.fill((18, 19, 22))

    # ── Header ───────────────────────────────────────────────────────────────
    title = font.render("TORCS Human Controller", True, (170, 172, 185))
    screen.blit(title, (12, 8))

    # ── Steering bar ─────────────────────────────────────────────────────────
    BW = W - 24
    pygame.draw.rect(screen, (40, 42, 50), (12, 38, BW, 14), border_radius=3)
    cx  = 12 + BW // 2
    bar_x = cx if fs >= 0 else cx + int(fs * (BW // 2))
    bar_w = abs(int(fs * (BW // 2)))
    pygame.draw.rect(screen, (80, 160, 255), (bar_x, 38, bar_w, 14), border_radius=3)
    pygame.draw.line(screen, (200, 200, 220), (cx, 34), (cx, 55), 2)
    lbl = small.render(f"STEER {fs:+.2f}", True, (140, 170, 220))
    screen.blit(lbl, (12, 54))

    # ── Accel bar ────────────────────────────────────────────────────────────
    _bar(screen, 12, 72, BW, 12, fa, (60, 200, 90))
    al = small.render(f"ACCEL {fa:.2f}  (raw {ctrl.accel:.2f})", True, (80, 200, 100))
    screen.blit(al, (12, 86))

    # ── Brake bar ────────────────────────────────────────────────────────────
    _bar(screen, 12, 104, BW, 12, fb, (200, 70, 70))
    bl = small.render(f"BRAKE {fb:.2f}  (raw {ctrl.brake:.2f})", True, (210, 90, 90))
    screen.blit(bl, (12, 118))

    # ── Slip indicator ───────────────────────────────────────────────────────
    pygame.draw.rect(screen, (35, 37, 45), (12, 138, BW, 10), border_radius=3)
    slip_w = int(ctrl.slip_level * BW)
    sc_col = (220, 80, 30) if ctrl.is_sc_active else (90, 95, 110)
    pygame.draw.rect(screen, sc_col, (12, 138, slip_w, 10), border_radius=3)
    sc_txt = small.render(
        f"SC {'AKTYWNY' if ctrl.is_sc_active else 'off'}  slip={ctrl.slip_level:.2f}",
        True, (220, 110, 60) if ctrl.is_sc_active else (90, 95, 110))
    screen.blit(sc_txt, (12, 150))

    # ── Launch control indicator ─────────────────────────────────────────────
    lc_on = speed_x_raw < 20.0
    lc_col = (220, 190, 30) if lc_on else (60, 63, 75)
    lc_txt = small.render(f"LC {'AKTYWNY' if lc_on else 'off'}  v={speed_x:.1f} km/h", True, lc_col)
    screen.blit(lc_txt, (12, 164))

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = [
        (f"Dystans : {dist:7.0f} m",   (140, 142, 155)),
        (f"Okrążenie: {lap}",          (140, 142, 155)),
        (f"Próbki  : {samples:6d}",    (140, 142, 155)),
    ]
    for i, (s, c) in enumerate(stats):
        img = small.render(s, True, c)
        screen.blit(img, (12, 184 + i * 18))

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = small.render("WASD | F8=zapis+reset | F9=koniec | BACKSPACE=usun probe", True, (60, 63, 78))
    screen.blit(leg, (12, H - 22))

    pygame.display.flip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  JSON APPEND HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _append_to_json_file(filepath, chunk):
    if not chunk: return
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            with open(filepath, 'w') as f:
                json.dump(chunk, f)
        else:
            with open(filepath, 'r+') as f:
                f.seek(0, 2)
                pos = f.tell()
                found = False
                while pos > 0:
                    pos -= 1
                    f.seek(pos)
                    if f.read(1) == ']':
                        found = True
                        break
                if found:
                    f.seek(pos)
                    f.truncate()
                    pos -= 1
                    is_empty_list = False
                    while pos >= 0:
                        f.seek(pos)
                        char = f.read(1)
                        if not char.isspace():
                            if char == '[': is_empty_list = True
                            break
                        pos -= 1
                    f.seek(0, 2)
                    inner_str = json.dumps(chunk)[1:-1]
                    if not is_empty_list and inner_str.strip():
                        f.write(",\n")
                    f.write(inner_str)
                    f.write("\n]")
    except Exception as e:
        print(f"  [Blad zapisu pliku JSON: {e}]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN DATA COLLECTION LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _truncate_json_file(filepath, keep_count):
    """Keep only the first keep_count records in a JSON list file."""
    if keep_count < 0:
        keep_count = 0
    if not os.path.exists(filepath):
        return 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  [Blad kasowania: {filepath} nie jest lista JSON]")
            return 0
        original_count = len(data)
        if keep_count >= original_count:
            return 0
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data[:keep_count], f)
        return original_count - keep_count
    except Exception as e:
        print(f"  [Blad kasowania probek z {filepath}: {e}]")
        return 0


def human_collect_data(num_laps: int = 50, max_steps: int = 500_000,
                       output_file: str = None,
                       restart_on_save: bool = False,
                       allow_low_progress: bool = False,
                       auto_reset_each_lap: bool = False):
    print("\n" + "=" * 62)
    print("  FAZA 1: Reczne zbieranie danych")
    print(f"  Cel: {num_laps} okrazen")
    print("  Uruchom TORCS z torem Corkscrew!")
    print("  KLIKNIJ OKNO PYGAME aby aktywowac sterowanie.")
    print("  Klawisze: W/Gora Gaz  S/Dol Hamulec  A/Lewo D/Prawo Skret")
    print("  F8 = zapisz segment i zresetuj auto w trybie korekt")
    print("  F9 = zapisz i wyjdz")
    print("  BACKSPACE = usun aktualny przejazd od ostatniego startu/mety i resetuj")
    if auto_reset_each_lap:
        print("  Auto-reset po mecie: zapis okrazenia i powrot na start.")
    print("=" * 62)

    pygame.init()
    screen = pygame.display.set_mode((440, 272))
    pygame.display.set_caption("TORCS Human Controller")
    font  = pygame.font.SysFont("monospace", 15, bold=True)
    small = pygame.font.SysFont("monospace", 13)

    C = snakeoil3.Client(p=PORT)
    C.MAX_STEPS = max_steps
    C.get_servers_input()
    S = C.S.d
    lap_start_dist = float(S.get('distRaced', 0.0))

    DATA_FILE_TEMP = "driving_data_toCombine.json"
    total_samples = 0
    
    def count_samples(filepath):
        count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    count += chunk.count('"state"')
        except Exception:
            pass
        return count

    def get_first_state_dim(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data:
                return len(data[0].get('state', []))
        except Exception:
            pass
        return None

    if output_file:
        OUTPUT_FILE = output_file
        print("  Cel korekt: nagrywaj trackPos 0.5-0.9, ratowanie z krawedzi,")
        print("  wolniejszy wjazd w zakret i lekkie bledy linii z bezpiecznym powrotem.")
        if os.path.exists(OUTPUT_FILE):
            state_dim = get_first_state_dim(OUTPUT_FILE)
            if state_dim is not None and state_dim not in (STATE_DIM, LEGACY_STATE_DIM):
                print(f"  NIEZGODNE DANE: {OUTPUT_FILE} ma state_dim={state_dim}, a kod oczekuje {STATE_DIM}.")
                print("  Przenies/usun stary plik przed dalsza kolekcja.")
                pygame.quit()
                C.shutdown()
                return
            samples_existing = count_samples(OUTPUT_FILE)
            total_samples += samples_existing
            print(f"  Tryb korekt: znaleziono {OUTPUT_FILE} ze {samples_existing} probkami.")
        else:
            print(f"  Tryb korekt: utworze nowy plik {OUTPUT_FILE}.")
    elif os.path.exists(DATA_FILE):
        state_dim = get_first_state_dim(DATA_FILE)
        if state_dim is not None and state_dim not in (STATE_DIM, LEGACY_STATE_DIM):
            print(f"  NIEZGODNE DANE: {DATA_FILE} ma state_dim={state_dim}, a kod oczekuje {STATE_DIM}.")
            print("  Przenies/usun stare driving_data*.json i zbierz dane od nowa.")
            pygame.quit()
            C.shutdown()
            return
        samples_main = count_samples(DATA_FILE)
        total_samples += samples_main
        print(f"  Znaleziono {DATA_FILE} ze {samples_main} probkami.")
        OUTPUT_FILE = DATA_FILE_TEMP
    else:
        OUTPUT_FILE = DATA_FILE

    if OUTPUT_FILE == DATA_FILE_TEMP and os.path.exists(DATA_FILE_TEMP):
        state_dim = get_first_state_dim(DATA_FILE_TEMP)
        if state_dim is not None and state_dim not in (STATE_DIM, LEGACY_STATE_DIM):
            print(f"  NIEZGODNE DANE: {DATA_FILE_TEMP} ma state_dim={state_dim}, a kod oczekuje {STATE_DIM}.")
            print("  Przenies/usun stary plik tymczasowy przed dalsza kolekcja.")
            pygame.quit()
            C.shutdown()
            return
        samples_temp = count_samples(DATA_FILE_TEMP)
        if samples_temp > 0:
            total_samples += samples_temp
            print(f"  Znaleziono wczesniejszy {DATA_FILE_TEMP}. Łącznie mamy {total_samples} probek.")

    print(f"  => ZAPIS TEJ SESJI TRAFI DO: {OUTPUT_FILE}")

    output_file_existing_count = count_samples(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
    session_data = []
    session_saved_count = 0
    lap_start_session_total = 0
    save_threads = []

    ctrl          = SmoothCarController()
    key_input     = KeyboardInput()
    running       = True
    lap_count     = 0
    prev_last_lap = 0.0
    f8_was_down   = False
    discard_was_down  = False

    old_sigint_handler = signal.getsignal(signal.SIGINT)

    def _request_stop(signum, frame):
        nonlocal running
        running = False
        print("\n  [Ctrl+C] Zatrzymuje i zapisuje zebrane dane...")

    signal.signal(signal.SIGINT, _request_stop)

    if key_input.uses_global_keys:
        print("  Globalny odczyt klawiszy aktywny: okno pygame nie musi miec fokusu.")
    else:
        print("  Uzywam odczytu pygame: okno pygame musi miec fokus.")
    print("  Filtry odrzucania probek sa wylaczone: zapisuje kazda klatke.")
    if allow_low_progress:
        print("  Uwaga: --allow-start nie zmienia nic, bo filtry probek sa wylaczone.")
    if auto_reset_each_lap:
        print("  Tryb okrazen: po kazdym pelnym okrazeniu auto wraca na linie startu.")

    def current_session_total() -> int:
        return session_saved_count + len(session_data)

    def wait_for_saves():
        if not save_threads:
            return
        alive = []
        for thread in save_threads:
            if thread.is_alive():
                thread.join()
            if thread.is_alive():
                alive.append(thread)
        save_threads[:] = alive

    def save_current_segment(label: str) -> int:
        nonlocal session_saved_count
        wait_for_saves()
        if not session_data:
            print(f"\n  [{label}] Brak nowych probek w segmencie.")
            return 0

        chunk_len = len(session_data)
        _append_to_json_file(OUTPUT_FILE, session_data)
        session_saved_count += chunk_len
        session_data.clear()
        print(f"\n  [{label}] Zapisano segment: {chunk_len} probek -> {OUTPUT_FILE}")
        return chunk_len

    def discard_current_pass(label: str) -> int:
        nonlocal session_data, session_saved_count
        wait_for_saves()

        before_total = current_session_total()
        if before_total <= lap_start_session_total:
            print(f"\n  [{label}] Brak probek do usuniecia w aktualnym przejezdzie.")
            return 0

        saved_keep_session = min(session_saved_count, lap_start_session_total)
        unsaved_keep = max(0, lap_start_session_total - saved_keep_session)
        file_keep_total = output_file_existing_count + saved_keep_session

        removed_from_file = _truncate_json_file(OUTPUT_FILE, file_keep_total)
        session_saved_count = saved_keep_session
        session_data = session_data[:unsaved_keep]

        removed_total = before_total - current_session_total()
        print(
            f"\n  [{label}] Usunieto aktualny przejazd: {removed_total} probek "
            f"({removed_from_file} z pliku) -> powrot do {OUTPUT_FILE}"
        )
        return removed_total

    def reset_car_to_start(label: str):
        nonlocal C, S, ctrl, prev_last_lap, lap_start_dist, lap_start_session_total

        C.R.d['steer'] = 0.0
        C.R.d['accel'] = 0.0
        C.R.d['brake'] = 0.0
        C.R.d['gear'] = 1
        C.R.d['meta'] = True
        C.respond_to_server()

        C.shutdown()
        time.sleep(0.5)
        C = snakeoil3.Client(p=PORT)
        C.MAX_STEPS = max_steps
        C.get_servers_input()
        S = C.S.d
        lap_start_dist = float(S.get('distRaced', 0.0))
        lap_start_session_total = current_session_total()

        ctrl = SmoothCarController()
        prev_last_lap = 0.0
        print(f"  [{label}] Reset auta i ponowne polaczenie gotowe.")

    for step in range(max_steps, 0, -1):
        # ── 1. Pobierz nowy stan z TORCS ─────────────────────────────────────
        C.get_servers_input()
        S = C.S.d

        speed_x   = float(S.get('speedX',    0.0))
        angle     = float(S.get('angle',     0.0))
        track_pos = float(S.get('trackPos',  0.0))
        dist      = float(S.get('distRaced', 0.0))

        # ── 2. Obsługa zdarzeń pygame ─────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        f8_down = key_input.pressed(pygame.K_F8)
        if f8_down and not f8_was_down:
            if restart_on_save:
                save_current_segment("F8")
                reset_car_to_start("F8")
                f8_was_down = True
                print("  [F8] Nagrywaj kolejny segment korekt.")
                continue
            running = False
        f8_was_down = f8_down
        discard_down = key_input.pressed(pygame.K_BACKSPACE)
        if discard_down and not discard_was_down:
            discard_current_pass("BACKSPACE")
            reset_car_to_start("BACKSPACE")
            discard_was_down = True
            print("  [BACKSPACE] Aktualny przejazd odrzucony. Nagrywaj probe ponownie.")
            continue
        discard_was_down = discard_down
        if key_input.pressed(pygame.K_F9):
            running = False

        if not running:
            break

        # ── 3. Oblicz wyjście kontrolera ─────────────────────────────────────
        final_steer, final_accel, final_brake = ctrl.update(key_input, speed_x, angle)

        # ── 4. Aktualizuj UI ─────────────────────────────────────────────────
        cur_step = max_steps - step
        current_total = total_samples + session_saved_count + len(session_data)
        draw_ui(screen, font, small, ctrl,
                final_steer, final_accel, final_brake,
                speed_x, dist, current_total, lap_count, speed_x)

        # ── 5. Zapisz dane ────────────────────────────────────────────────────
        #  Zapisujemy tylko gdy auto jest aktywne i znajduje się NA torze.
        #  trackPos ∈ (-1, 1) oznacza bycie na asfalcie.
        state = get_state(S, lap_start_dist)
        session_data.append({
            'state': state.tolist(),
            'action': [final_steer, final_accel, final_brake],
        })
        current_total += 1

        # ── 6. Logowanie ──────────────────────────────────────────────────────
        if cur_step % 500 == 0 and cur_step > 0:
            print(f"  krok {cur_step:5d} | dist={dist:6.0f}m | "
                  f"v={speed_x:5.1f}km/h | tpos={track_pos:+.3f} | "
                  f"SC={'ON ' if ctrl.is_sc_active else 'off'} | "
                  f"probki={current_total}")

        if cur_step % 2000 == 0 and cur_step > 0:
            import threading
            chunk_to_save = session_data[:]
            session_saved_count += len(chunk_to_save)
            session_data.clear()
            
            def bg_save(chunk):
                _append_to_json_file(OUTPUT_FILE, chunk)
                print(f"\n  [Auto-zapis w tle: dopisano {len(chunk)} nowych probek do {OUTPUT_FILE}]")
                
            thread = threading.Thread(target=bg_save, args=(chunk_to_save,), daemon=True)
            save_threads.append(thread)
            thread.start()

        # ── 7. Sprawdź okrążenie ──────────────────────────────────────────────
        new_lap_completed = False
        last_lap = float(S.get('lastLapTime', 0.0))
        if last_lap > 0 and last_lap != prev_last_lap:
            lap_count    += 1
            prev_last_lap = last_lap
            new_lap_completed = True
            if auto_reset_each_lap and lap_count < num_laps:
                print(f"\n  Okrazenie {lap_count} ukonczone! Czas: {last_lap:.2f}s | Dist: {dist:.0f}m")
                save_current_segment(f"Lap {lap_count}")
                reset_car_to_start(f"Lap {lap_count}")
                continue
            print(f"\n  Okrazenie {lap_count} ukonczone! "
                  f"Czas: {last_lap:.2f}s | Dist: {dist:.0f}m | "
                  f"Probki: {current_total}")
            if lap_count >= num_laps:
                print(f"  Zebrano {num_laps} okrazen — zatrzymuje.")
                break

        # ── 8. Wyślij komendy do TORCS ────────────────────────────────────────
        if new_lap_completed:
            lap_start_session_total = current_session_total()
            lap_start_dist = dist

        R = C.R.d
        R['steer'] = final_steer
        R['accel'] = final_accel
        R['brake'] = final_brake
        _apply_gears(speed_x, R)
        C.respond_to_server()

    pygame.quit()
    C.shutdown()
    signal.signal(signal.SIGINT, old_sigint_handler)

    wait_for_saves()
    if session_data:
        _append_to_json_file(OUTPUT_FILE, session_data)
        session_saved_count += len(session_data)
        
    print(f"\n  Zapisano sesję. Łączna liczba probek w obu plikach → {total_samples + session_saved_count}")
    if output_file:
        print(f"  Po sesji korekt uruchom: python combine_corrections.py")
    if OUTPUT_FILE == "driving_data_toCombine.json":
        print(f"  PAMIĘTAJ: Po zakończeniu wszystkich przejazdów użyj komendy:")
        print(f"            python combine_data.py")
        print(f"            aby połączyć zebrane próbki w jeden główny plik driving_data.json!")


if __name__ == "__main__":
    human_collect_data()
