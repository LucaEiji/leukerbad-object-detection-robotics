"""
path_planner.py : planifie le meilleur chemin du bras vers une puff (x, y, z, orientation).

Entrée  : pose de la puff + position de départ (angles des moteurs).
Sortie  : un plan = liste d'étapes (approche, descente verticale, fermeture pince, levée, retour)
          avec les angles de chaque moteur à chaque point. Rien n'est exécuté tant que tu
          n'appelles pas execute(plan).

Repère de la puff : MEME repère que arm_motion.move_xy -> cm relatifs à l'ORIGINE (pose de
repos). +x = avant (s'éloigne de la base), +y = gauche.
  z_cm    : hauteur du centre de la puff au-dessus de la table (cm)
  yaw_deg : orientation du grand axe de la puff dans le plan de la table, mesurée depuis +x
            vers +y (sens anti-horaire vu de dessus).

Choix du "meilleur" chemin = le plus rapide (mouvement articulaire minimal) PARMI les chemins
valides : limites articulaires respectées, pas de collision avec la table (coude, poignet et
pince au-dessus de la table en transit), descente strictement verticale pour saisir.

Usage :
    import arm_motion as am, path_planner as pp
    am.connect(dry_run=True)
    plan = pp.plan_pick(pp.Puff(5.0, -4.0, 1.0, 30.0), start_joints=am.read_joints())
    pp.print_plan(plan)
    pp.execute(plan)           # en dry run : rien ne bouge
Test visuel : python path_planner.py   (écrit plan_result.png)
"""
import math
from dataclasses import dataclass

import arm_motion as am

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
HOVER_CM = 6.0            # hauteur d'approche au-dessus de la puff (cm)
SAFE_CM = 12.0            # hauteur de transit sûre (cm)
CLEARANCE_CM = 2.0        # le coude/poignet/pince ne descendent jamais sous cette hauteur en transit
DESCENT_STEPS = 8         # nombre de points de la descente verticale
TRANSIT_SAMPLES = 40      # points échantillonnés pour vérifier la collision
MAX_JUMP_DEG = 25.0       # un saut articulaire plus grand entre 2 points de la descente = solution invalide

# Orientation de la pince (wrist_roll). A CALIBRER sur le vrai bras :
#   roll = ROLL_OFFSET + ROLL_SIGN * (angle_mâchoires_monde - pan - JAW_AXIS_AT_ZERO_DEG)
ROLL_SIGN = 1.0
ROLL_OFFSET = 0.0                  # valeur de wrist_roll quand les mâchoires sont dans la position "zéro"
JAW_AXIS_AT_ZERO_DEG = 90.0        # axe de fermeture vs direction radiale au roll "zéro" (90 = perpendiculaire)
ROLL_LIMITS = (-100.0, 100.0)
GRASP_ACROSS = True                # True : les mâchoires se ferment en travers du grand axe de la puff

SHORT = {"shoulder_pan.pos": "pan", "shoulder_lift.pos": "lift", "elbow_flex.pos": "coude",
         "wrist_flex.pos": "poignet", "wrist_roll.pos": "roll"}
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]


@dataclass
class Puff:
    x_cm: float
    y_cm: float
    z_cm: float
    yaw_deg: float


# ----------------------------------------------------------------------------
# OUTILS GEOMETRIQUES
# ----------------------------------------------------------------------------
def _heights_and_tip(q):
    """FK détaillée : (z_coude, z_poignet, z_pince, x, y, r) en mètres, au-dessus de la table."""
    th1 = math.radians((q["shoulder_pan.pos"] - am.PAN_OFFSET) / am.PAN_SIGN)
    a2 = math.radians((q["shoulder_lift.pos"] - am.LIFT_OFFSET) / am.LIFT_SIGN)
    a3 = a2 + math.radians((q["elbow_flex.pos"] - am.ELBOW_OFFSET) / am.ELBOW_SIGN)
    a4 = a3 + math.radians((q["wrist_flex.pos"] - am.WRIST_OFFSET) / am.WRIST_SIGN)
    ze = am.SHOULDER_Z + am.L1 * math.sin(a2) - am.TABLE_Z
    zw = ze + am.L2 * math.sin(a3)
    zt = zw + am.L_TOOL * math.sin(a4)
    r = am.L1 * math.cos(a2) + am.L2 * math.cos(a3) + am.L_TOOL * math.cos(a4)
    return ze, zw, zt, r * math.cos(th1), r * math.sin(th1), r


def _lerp(q0, q1, a):
    return {k: q0[k] + (q1[k] - q0[k]) * a for k in JOINTS}


def _dist(q0, q1):
    return max(abs(q0[k] - q1[k]) for k in JOINTS)


def _segment_time(qs):
    return sum(_dist(a, b) for a, b in zip(qs, qs[1:])) / am.SPEED_DEG_S


def _collision_free(q0, q1):
    """Interpolation articulaire q0->q1 : coude, poignet et pince restent au-dessus de la table."""
    clr = CLEARANCE_CM / 100.0
    for i in range(TRANSIT_SAMPLES + 1):
        q = _lerp(q0, q1, i / TRANSIT_SAMPLES)
        ze, zw, zt, *_ = _heights_and_tip(q)
        if min(ze, zw, zt) < clr:
            return False
    return True


def _roll_for(puff_yaw_deg, pan_world_deg, roll_ref):
    """Wrist_roll qui oriente les mâchoires (en travers de la puff). Modulo 180 : choisit le plus proche de roll_ref."""
    jaw_yaw = puff_yaw_deg + (90.0 if GRASP_ACROSS else 0.0)
    base = ROLL_OFFSET + ROLL_SIGN * (jaw_yaw - pan_world_deg - JAW_AXIS_AT_ZERO_DEG)
    cands = []
    for k in range(-4, 5):
        r = base + 180.0 * k
        if ROLL_LIMITS[0] <= r <= ROLL_LIMITS[1]:
            cands.append(r)
    if not cands:
        raise ValueError("Orientation de la puff impossible avec les limites de wrist_roll")
    return min(cands, key=lambda r: abs(r - roll_ref))


def _puff_world(puff):
    """Puff (repère relatif origine, cm) -> (x, y, z_m, yaw_monde_deg) dans le repère base."""
    if am._origin is None:
        am._init_origin()
    x0, y0, _ = am._origin
    x = x0 + am.AXIS_X_SIGN * am._mm(puff.x_cm)
    y = y0 + am.AXIS_Y_SIGN * am._mm(puff.y_cm)
    yaw = math.degrees(math.atan2(am.AXIS_Y_SIGN * math.sin(math.radians(puff.yaw_deg)),
                                  am.AXIS_X_SIGN * math.cos(math.radians(puff.yaw_deg))))
    return x, y, max(am._mm(puff.z_cm), am.MIN_Z), yaw


def _ik_with_roll(x, y, z, roll):
    q = am.ik_xy(x, y, z)
    q["wrist_roll.pos"] = roll
    return q


# ----------------------------------------------------------------------------
# PLANIFICATION
# ----------------------------------------------------------------------------
def plan_pick(puff, start_joints=None, end_joints=None):
    """
    Renvoie un plan (dict). Lève ValueError si aucune solution valide (cible inatteignable,
    descente instable, collision partout...). Aucun mouvement n'est effectué.
    """
    if start_joints is None:
        start_joints = am.read_joints()
    start = {k: float(start_joints[k]) for k in JOINTS}
    end = dict(start) if end_joints is None else {k: float(end_joints[k]) for k in JOINTS}

    x, y, z_grasp, yaw_world = _puff_world(puff)
    pan_world = math.degrees(math.atan2(y, x))
    roll = _roll_for(yaw_world, pan_world, start["wrist_roll.pos"])

    # 1. descente verticale : points (x, y) fixes, z de hover à grasp ; IK à chaque point
    z_hover = z_grasp + HOVER_CM / 100.0
    descent = []
    for i in range(DESCENT_STEPS + 1):
        z = z_hover + (z_grasp - z_hover) * i / DESCENT_STEPS
        descent.append(_ik_with_roll(x, y, z, roll))
    for a, b in zip(descent, descent[1:]):
        if _dist(a, b) > MAX_JUMP_DEG:
            raise ValueError("Descente verticale instable (changement de configuration du bras) : cible trop proche des limites")
    hover, grasp = descent[0], descent[-1]

    # 2. chemins de transit candidats : start -> hover
    candidates = []
    if _collision_free(start, hover):
        candidates.append(("direct", [start, hover]))
    _, _, szt, sx, sy, _ = _heights_and_tip(start)
    z_via = max(szt, SAFE_CM / 100.0)
    try:
        via_start = _ik_with_roll(sx, sy, z_via, start["wrist_roll.pos"])
        via_target = _ik_with_roll(x, y, max(z_via, z_hover), roll)
        if (_collision_free(start, via_start) and _collision_free(via_start, via_target)
                and _collision_free(via_target, hover)):
            candidates.append(("par_hauteur_sure", [start, via_start, via_target, hover]))
    except ValueError:
        pass
    if not candidates:
        raise ValueError("Aucun chemin sans collision depuis la position de départ vers la puff")
    best_name, best_pts = min(candidates, key=lambda c: _segment_time(c[1]))

    # 3. retour : levée verticale puis chemin vers end
    lift = list(reversed(descent))
    ret_pts = [hover, end]
    if not _collision_free(hover, end):
        try:
            _, _, _, hx, hy, _ = _heights_and_tip(hover)
            via = _ik_with_roll(hx, hy, max(SAFE_CM / 100.0, z_hover), hover["wrist_roll.pos"])
            ret_pts = [hover, via, end]
        except ValueError:
            pass
        if not all(_collision_free(a, b) for a, b in zip(ret_pts, ret_pts[1:])):
            raise ValueError("Pas de chemin de retour sans collision")

    steps = [
        dict(name=f"transit ({best_name})", path=best_pts, gripper=am.GRIPPER_OPEN),
        dict(name="descente verticale", path=descent, gripper=None),
        dict(name="fermeture pince", path=[grasp], gripper=am.GRIPPER_CLOSED),
        dict(name="levée verticale", path=lift, gripper=None),
        dict(name="retour", path=ret_pts, gripper=None),
    ]
    total = sum(_segment_time(s["path"]) for s in steps)
    return dict(steps=steps, candidates={n: _segment_time(p) for n, p in candidates},
                chosen=best_name, est_time_s=total, puff_world=(x, y, z_grasp, yaw_world), puff_yaw_rel=puff.yaw_deg, roll=roll)


def print_plan(plan):
    x, y, z, yaw = plan["puff_world"]
    print(f"Puff (repère base) : x={x*100:.1f} y={y*100:.1f} z={z*100:.1f} cm, yaw monde {yaw:.0f} deg "
          f"-> wrist_roll = {plan['roll']:.1f}")
    print("Chemins de transit évalués (temps estimé) :",
          ", ".join(f"{n}: {t:.1f} s" for n, t in plan["candidates"].items()), f"-> choisi : {plan['chosen']}")
    for s in plan["steps"]:
        print(f"\n[{s['name']}]" + (f"  pince -> {s['gripper']}" if s["gripper"] is not None else ""))
        for q in s["path"]:
            print("   " + "  ".join(f"{SHORT[k]}={q[k]:7.2f}" for k in JOINTS))
    print(f"\nDurée estimée : {plan['est_time_s']:.1f} s")


def execute(plan, confirm=True):
    """Exécute le plan via arm_motion. Rien ne bouge si arm_motion est en dry run."""
    for s in plan["steps"]:
        if confirm and not am._dry_run:
            if input(f"Exécuter l'étape « {s['name']} » ? [Entrée = oui, q = stop] ").strip().lower() == "q":
                print("Arrêt demandé.")
                return False
        if s["gripper"] is not None:
            am.set_gripper(s["gripper"])
        for q in s["path"]:
            am._move(dict(q, **{"gripper.pos": am._state["gripper.pos"]}))
    return True


# ----------------------------------------------------------------------------
# TEST VISUEL (aucun robot)
# ----------------------------------------------------------------------------
def plot_plan(plan, path="plan_result.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, y0, _ = am._origin
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for s, c in zip(plan["steps"], colors):
        pts = []
        for q0, q1 in zip(s["path"], s["path"][1:]) if len(s["path"]) > 1 else []:
            for i in range(21):
                pts.append(_heights_and_tip(_lerp(q0, q1, i / 20)))
        if not pts:
            pts = [_heights_and_tip(s["path"][0])]
        xs = [am.AXIS_Y_SIGN * (p[4] - y0) * 100 for p in pts]
        ys = [am.AXIS_X_SIGN * (p[3] - x0) * 100 for p in pts]
        a1.plot(xs, ys, "-o", ms=2, color=c, label=s["name"])
        rr = [p[5] * 100 for p in pts]
        a2.plot(rr, [p[2] * 100 for p in pts], "-o", ms=2, color=c, label="pince : " + s["name"])
        a2.plot(rr, [p[1] * 100 for p in pts], ":", color=c, alpha=0.6)
    px, py, pz, yaw = plan["puff_world"]
    ux, uy = am.AXIS_Y_SIGN * (py - y0) * 100, am.AXIS_X_SIGN * (px - x0) * 100
    a1.plot(ux, uy, "k*", ms=16, label="puff")
    # orientation de la puff : segment le long de son grand axe (en repère affichage dy, dx)
    ang = math.radians(plan["puff_yaw_rel"])
    L = 2.5
    dxs, dys = L * math.cos(ang), L * math.sin(ang)
    a1.plot([ux - dys, ux + dys], [uy - dxs, uy + dxs], "k-", lw=4, alpha=0.5, label="grand axe de la puff")
    a1.set_title("Vue de dessus (cm, relatif à l'origine) : trajet du bout de pince")
    a1.set_xlabel("dy (cm)  gauche <-> droite") 
    a1.set_ylabel("dx (cm)  avant")
    a1.invert_xaxis()
    a1.grid(True, alpha=0.4)
    a1.set_aspect("equal")
    a1.legend(fontsize=7)
    a2.axhline(CLEARANCE_CM, color="r", ls="--", lw=1, label="hauteur mini en transit")
    a2.axhline(0, color="k", lw=2)
    a2.set_title("Vue de côté : hauteur de la pince (trait plein) et du poignet (pointillé)")
    a2.set_xlabel("distance depuis l'axe de la base (cm)")
    a2.set_ylabel("hauteur au-dessus de la table (cm)")
    a2.grid(True, alpha=0.4)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("Écrit :", path)


if __name__ == "__main__":
    am.ORIGIN_OVERRIDE = (0.18, 0.0, 0.07)       # exemple : origine à 18 cm de la base
    am.connect(dry_run=True)
    start = am.ik_xy(0.18, 0.0, 0.12)            # départ : au-dessus de l'origine, 12 cm de haut
    plan = plan_pick(Puff(x_cm=5.0, y_cm=-4.0, z_cm=1.0, yaw_deg=30.0), start_joints=start)
    print_plan(plan)
    execute(plan)
    plot_plan(plan)