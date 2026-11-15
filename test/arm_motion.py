"""
arm_motion.py : déplacement du SO-101 vers un point (x, y) du plan de la table.

Module autonome (n'a pas besoin de pick_puff.py). Utilisation depuis le code global :

    import arm_motion as am

    am.connect(dry_run=True)          # dry_run=True : aucun mouvement, angles affichés seulement
    am.move_xy(3.5, -2.1)             # UNITES = cm, résolution 1 mm (0,1 cm)
                                      # = 3,5 cm vers l'avant, 2,1 cm vers la droite de l'ORIGINE
    am.move_xy(0, 0)                  # retour à l'origine (au-dessus du centre)
    am.move_xy(0, 5.0, z_cm=1.2)      # z_cm optionnel : hauteur de la pince au-dessus de la table
    am.set_gripper(am.GRIPPER_CLOSED)
    am.go_rest()                      # retour exact à la pose de repos (espace articulaire)
    am.disconnect()

REPERE : l'origine (0, 0) est le point de la table sous la pince quand le bras est dans
REST_POSE. Les axes sont parallèles à ceux de la base : +X = s'éloigner de la base (vers
l'avant), +Y = vers la gauche (vu depuis la base, regardant vers l'avant).
goto_xy(x_cm, y_cm) reste disponible en coordonnées absolues (repère base), aussi en cm.

PINCE : elle reste le plus verticale possible ; si la cible est trop loin elle s'incline,
au maximum de MAX_TILT_DEG (20 deg). Au-delà : ValueError, le bras ne bouge pas.
am.reach_map() affiche, sans bouger le bras, quels points de [-15 cm, +15 cm]^2 autour de
l'origine sont atteignables et avec quelle inclinaison de pince.

SECURITE : le mode réel (dry_run=False) n'est à utiliser qu'après vérification des
constantes de calibration ci-dessous, bras surveillé, arrêt d'urgence à portée.
"""
import math
import time

# ----------------------------------------------------------------------------
# CONFIGURATION (à calibrer)
# ----------------------------------------------------------------------------
ROBOT_PORT = "/dev/ttyACM0"        # Windows : "COM3"
ROBOT_ID = "my_so101"

# Pose de repos / scan (degrés, gripper 0-100) : à mesurer bras en couple désactivé
REST_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -100.0,
    "elbow_flex.pos": 100.0,
    "wrist_flex.pos": 60.0,
    "wrist_roll.pos": 0.0,
}

# Géométrie (m)
L1, L2 = 0.1159, 0.1350            # épaule->coude, coude->poignet
L_TOOL = 0.10                      # axe du poignet -> bout de la pince (pince verticale)
SHOULDER_Z = 0.10                  # hauteur de l'axe d'épaule au-dessus de la table
TABLE_Z = 0.0                      # hauteur de la table dans le repère robot

DEFAULT_Z = 0.05                   # hauteur de la pince par défaut (m au-dessus de la table)
MIN_Z = 0.005                      # garde-fou : jamais plus bas que 5 mm au-dessus de la table

# Conversion angle "maths" (deg) -> valeur robot : valeur = SIGN * angle + OFFSET
PAN_SIGN, PAN_OFFSET = 1.0, 0.0
LIFT_SIGN, LIFT_OFFSET = -1.0, 0.0
ELBOW_SIGN, ELBOW_OFFSET = 1.0, 0.0
WRIST_SIGN, WRIST_OFFSET = 1.0, 0.0

# Limites de sécurité sur les valeurs envoyées (degrés) : à ajuster à ton bras
JOINT_LIMITS = {
    "shoulder_pan.pos": (-110.0, 110.0),
    "shoulder_lift.pos": (-110.0, 110.0),
    "elbow_flex.pos": (-110.0, 110.0),
    "wrist_flex.pos": (-110.0, 110.0),
}

GRIPPER_OPEN = 40.0
GRIPPER_CLOSED = 0.0

STEP_HZ = 30
SPEED_DEG_S = 40.0                 # vitesse articulaire max (lent = sûr)


# ----------------------------------------------------------------------------
# CINEMATIQUE INVERSE
# ----------------------------------------------------------------------------
MAX_TILT_DEG = 20.0     # inclinaison max de la pince par rapport à la verticale (0 = toujours vers le bas)


def _ik_for_pitch(x, y, z, phi):
    """IK pour une inclinaison absolue de la pince phi (rad, -pi/2 = vers le bas). None si impossible."""
    r = math.hypot(x, y)
    rw = r - L_TOOL * math.cos(phi)                       # poignet : bout de pince reculé de L_TOOL
    hw = (TABLE_Z + z - SHOULDER_Z) - L_TOOL * math.sin(phi)
    d = math.hypot(rw, hw)
    if d > L1 + L2 - 1e-3 or d < abs(L1 - L2) + 1e-3:
        return None
    th3 = -math.acos(max(-1.0, min(1.0, (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2))))
    th2 = math.atan2(hw, rw) - math.atan2(L2 * math.sin(th3), L1 + L2 * math.cos(th3))
    th4 = phi - th2 - th3
    joints = {
        "shoulder_pan.pos": PAN_SIGN * math.degrees(math.atan2(y, x)) + PAN_OFFSET,
        "shoulder_lift.pos": LIFT_SIGN * math.degrees(th2) + LIFT_OFFSET,
        "elbow_flex.pos": ELBOW_SIGN * math.degrees(th3) + ELBOW_OFFSET,
        "wrist_flex.pos": WRIST_SIGN * math.degrees(th4) + WRIST_OFFSET,
        "wrist_roll.pos": REST_POSE["wrist_roll.pos"],
    }
    for k, (lo, hi) in JOINT_LIMITS.items():
        if not lo <= joints[k] <= hi:
            return None
    return joints


def ik_xy(x, y, z=DEFAULT_Z, return_tilt=False):
    """
    (x, y) en mètres (+ hauteur z au-dessus de la table) -> angles des moteurs.
    La pince est le plus verticale possible : on essaie d'abord vertical, puis on l'incline
    progressivement (jusqu'à MAX_TILT_DEG) si la cible est trop loin. Cela étend la portée
    d'environ L_TOOL. Lève ValueError si aucune inclinaison ne permet d'atteindre la cible.
    """
    if z < MIN_Z:
        raise ValueError(f"z={z:.3f} m trop bas (minimum {MIN_Z} m)")
    dev = 0.0
    while dev <= MAX_TILT_DEG + 1e-9:
        for s in ((1, -1) if dev else (1,)):
            joints = _ik_for_pitch(x, y, z, -math.pi / 2 + s * math.radians(dev))
            if joints is not None:
                return (joints, dev) if return_tilt else joints
        dev += 1.0
    raise ValueError(f"Cible inatteignable (x={x:.3f}, y={y:.3f}, z={z:.3f}, "
                     f"r={math.hypot(x, y):.3f} m) avec inclinaison <= {MAX_TILT_DEG:.0f} deg et les limites articulaires")


def reach_map(span_cm=15, step_cm=5, z_cm=None):
    """Teste en simulation la grille [-span, +span]^2 (cm) autour de l'origine et l'affiche.
    Chiffre = inclinaison de la pince en degrés ; '--' = inatteignable. Aucun mouvement."""
    if _origin is None:
        _init_origin()
    x0, y0, z0 = _origin
    zz = z0 if z_cm is None else _mm(z_cm)
    n = int(round(span_cm / step_cm))
    vals = [i * step_cm for i in range(-n, n + 1)]
    print(f"Portée autour de l'origine (z={zz*100:.1f} cm, inclinaison max {MAX_TILT_DEG:.0f} deg)")
    print("lignes = dx cm (avant en haut) ; colonnes = dy cm (gauche à gauche)")
    print("dx\\dy " + " ".join(f"{-v:+5.0f}" for v in vals))
    for dx in reversed(vals):
        row = []
        for dy in reversed(vals):
            try:
                _, tilt = ik_xy(x0 + AXIS_X_SIGN * _mm(dx), y0 + AXIS_Y_SIGN * _mm(dy), zz, return_tilt=True)
                row.append(f"{tilt:5.0f}")
            except ValueError:
                row.append("   --")
        print(f"{dx:+5.0f}  " + " ".join(row))


def _mm(v_cm):
    """cm -> mètres, arrondi au millimètre."""
    return round(float(v_cm) * 10.0) / 1000.0


def fk_xy(joints):
    """
    Cinématique directe : angles des moteurs -> (x, y, z) du bout de la pince dans le repère
    base (z au-dessus de la table). Inverse de ik_xy (mêmes SIGN/OFFSET).
    """
    th1 = math.radians((joints["shoulder_pan.pos"] - PAN_OFFSET) / PAN_SIGN)
    th2 = math.radians((joints["shoulder_lift.pos"] - LIFT_OFFSET) / LIFT_SIGN)
    th3 = math.radians((joints["elbow_flex.pos"] - ELBOW_OFFSET) / ELBOW_SIGN)
    th4 = math.radians((joints["wrist_flex.pos"] - WRIST_OFFSET) / WRIST_SIGN)
    a2, a3, a4 = th2, th2 + th3, th2 + th3 + th4
    r = L1 * math.cos(a2) + L2 * math.cos(a3) + L_TOOL * math.cos(a4)
    h = L1 * math.sin(a2) + L2 * math.sin(a3) + L_TOOL * math.sin(a4)
    return r * math.cos(th1), r * math.sin(th1), SHOULDER_Z + h - TABLE_Z


# ----------------------------------------------------------------------------
# ORIGINE = POSE DE REPOS
# ----------------------------------------------------------------------------
# Par défaut l'origine (x0, y0, z0) est calculée par cinématique directe à partir de REST_POSE.
# Si tes SIGN/OFFSET ne sont pas encore fiables, mesure plutôt à la règle la position du bout
# de la pince au repos dans le repère base (x0 devant l'axe de la base, y0 à gauche) et mets :
#   ORIGIN_OVERRIDE = (x0, y0)
ORIGIN_OVERRIDE = (0.10, 0.0, 0.02)

# Sens des axes relatifs (mets -1 pour inverser un axe si besoin)
AXIS_X_SIGN = 1.0
AXIS_Y_SIGN = 1.0

_origin = None      # (x0, y0, z0) dans le repère base


def _init_origin():
    global _origin
    x0, y0, z0 = fk_xy(REST_POSE)
    if ORIGIN_OVERRIDE is not None:
        x0, y0 = ORIGIN_OVERRIDE[:2]
        if len(ORIGIN_OVERRIDE) == 3:
            z0 = ORIGIN_OVERRIDE[2]
    _origin = (x0, y0, max(z0, MIN_Z))
    print(f"[arm_motion] Origine (repère base) : x0={x0:.3f} y0={y0:.3f} z0={z0:.3f} m"
          + (" [mesurée]" if ORIGIN_OVERRIDE is not None else " [calculée depuis REST_POSE]"))
    if not (MIN_Z <= z0 <= 0.25):
        print("[arm_motion] ATTENTION : z0 incohérent -> REST_POSE doit être une pose "
              "pince vers le bas au-dessus de la table, ou vérifie SIGN/OFFSET/L_TOOL.")


def max_symmetric_range(x0_cm, z_cm, y0_cm=0.0, limit_cm=30):
    """Plus grand R (cm, entier) tel que TOUT le carré [-R, +R]^2 autour de (x0, y0) soit atteignable."""
    best = -1
    for R in range(0, limit_cm + 1):
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                if max(abs(a), abs(b)) != R:
                    continue
                try:
                    ik_xy(_mm(x0_cm + a), _mm(y0_cm + b), _mm(z_cm))
                except ValueError:
                    return best
        best = R
    return best


def best_origin(z_cm=7.0, x_min_cm=8.0, x_max_cm=35.0, step_cm=0.5):
    """
    Cherche la distance x0 (cm, depuis l'axe de la base, sur l'axe avant) qui maximise la plage
    symétrique [-R, +R] atteignable autour de l'origine, à la hauteur z_cm.
    Renvoie (x0_cm, R_cm). Aucun mouvement.
    """
    best = (None, -1)
    x = x_min_cm
    while x <= x_max_cm + 1e-9:
        R = max_symmetric_range(x, z_cm)
        if R > best[1]:
            best = (x, R)
        x += step_cm
    return best


def use_best_origin(z_cm=7.0):
    """Fixe l'origine à l'endroit optimal. Il faut ensuite PLACER le bras physiquement
    au-dessus de ce point (distance x0 mesurée depuis l'axe de la base) avant de bouger."""
    global ORIGIN_OVERRIDE
    x0, R = best_origin(z_cm)
    if x0 is None or R < 0:
        raise RuntimeError("Aucune origine ne permet d'atteindre même (0, 0) : vérifier la géométrie.")
    ORIGIN_OVERRIDE = (x0 / 100.0, 0.0, z_cm / 100.0)
    _init_origin()
    print(f"[arm_motion] Origine optimale : à {x0:.1f} cm de l'axe de la base, hauteur {z_cm:.1f} cm "
          f"-> plage symétrique garantie : +/-{R} cm en X et en Y.")
    return x0, R


# ----------------------------------------------------------------------------
# PILOTAGE DU BRAS
# ----------------------------------------------------------------------------
_robot = None
_state = dict(REST_POSE, **{"gripper.pos": GRIPPER_CLOSED})
_dry_run = True


def connect(dry_run=False):
    """Connecte le bras (ou simule si dry_run=True : rien n'est envoyé au robot)."""
    global _robot, _state, _dry_run
    _dry_run = dry_run
    if dry_run:
        print("[arm_motion] DRY RUN : aucun mouvement ne sera envoyé au robot.")
        _init_origin()
        return
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    cfg = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, use_degrees=True)
    _robot = SO101Follower(cfg)
    _robot.connect()
    _state = read_joints()
    _init_origin()
    print("[arm_motion] Bras connecté. Place-le manuellement dans la pose de repos avant move_xy().")


def disconnect():
    global _robot
    if _robot is not None:
        _robot.disconnect()
        _robot = None


def read_joints():
    """Angles actuels des moteurs (dict). En dry run : dernière consigne."""
    if _robot is None:
        return dict(_state)
    obs = _robot.get_observation()
    return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}


def _move(target):
    """Mouvement interpolé et lissé en espace articulaire vers 'target' (dict complet)."""
    global _state
    start = dict(_state)
    dist = max(abs(target[k] - start[k]) for k in target)
    steps = max(1, int(dist / SPEED_DEG_S * STEP_HZ))
    for i in range(1, steps + 1):
        a = i / steps
        a = a * a * (3 - 2 * a)
        cmd = {k: start[k] + (target[k] - start[k]) * a for k in target}
        if _robot is not None:
            _robot.send_action(cmd)
        time.sleep(1.0 / STEP_HZ if _robot is not None else 0)
    _state = dict(target)


def goto_xy(x_cm, y_cm, z_cm=None):
    """
    Déplace la pince au point ABSOLU (x, y) du repère base, en cm (résolution 1 mm).
    z_cm : hauteur de la pince au-dessus de la table (None = DEFAULT_Z).
    Renvoie le dict des positions de chaque moteur atteintes.
    Lève ValueError (sans bouger) si la cible est inatteignable.
    """
    x, y = _mm(x_cm), _mm(y_cm)
    z = DEFAULT_Z if z_cm is None else _mm(z_cm)
    joints, tilt = ik_xy(x, y, z, return_tilt=True)   # calcul d'abord : aucune action si erreur
    target = dict(joints, **{"gripper.pos": _state["gripper.pos"]})
    _move(target)
    print(f"[arm_motion] -> x={x*100:.1f} y={y*100:.1f} z={z*100:.1f} cm, inclinaison {tilt:.0f} deg | "
          + "  ".join(f"{k.split('.')[0]}={v:7.2f}" for k, v in target.items()))
    return target


def move_xy(dx_cm, dy_cm, z_cm=None):
    """
    Déplace la pince au point (dx, dy) en cm (résolution 1 mm) PAR RAPPORT A L'ORIGINE
    (= pose de repos). (0, 0) = au-dessus du point de départ.
    +dx = vers l'avant, +dy = vers la gauche.
    z_cm : hauteur de la pince au-dessus de la table (None = hauteur de la pose de repos).
    Renvoie le dict des positions de chaque moteur. ValueError (sans bouger) si inatteignable.
    """
    if _origin is None:
        raise RuntimeError("Appelle connect() avant move_xy().")
    x0, y0, z0 = _origin
    x = x0 + AXIS_X_SIGN * _mm(dx_cm)
    y = y0 + AXIS_Y_SIGN * _mm(dy_cm)
    return goto_xy(x * 100, y * 100, z0 * 100 if z_cm is None else z_cm)


def set_gripper(value, wait=0.8):
    """Ouvre/ferme la pince (0 = fermée, ~40 = ouverte), puis attend."""
    target = dict(_state, **{"gripper.pos": value})
    _move(target)
    time.sleep(wait if _robot is not None else 0)


def go_rest():
    """Retour à la pose de repos / scan (espace articulaire)."""
    _move(dict(REST_POSE, **{"gripper.pos": _state["gripper.pos"]}))


# ----------------------------------------------------------------------------
# DEMO (dry run) : python arm_motion.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    connect(dry_run=True)
    reach_map()
    for dx, dy in [(0, 0), (3.5, 0), (0, 5.2), (0, -5.2), (-3.0, 0), (15, 0)]:
        try:
            move_xy(dx, dy)
        except ValueError as e:
            print("Refusé :", e)
    go_rest()
    disconnect()