"""
sim_view.py : visualisation du code de arm_motion.py SANS robot (dry run uniquement).

Le bout de la pince est un point sur un quadrillage (cm, repère relatif à l'origine = repos).
  - fond vert  : zone atteignable par le code (avec l'inclinaison de pince en teinte)
  - croix verte: cible atteinte ; croix rouge : cible REFUSEE (hors portée / hors limites)
  - ligne      : trajet réel du bout de pince (interpolation articulaire puis cinématique directe)
Le terminal affiche aussi, pour chaque test, l'erreur entre la cible demandée et la position
retrouvée par cinématique directe (doit être ~0 mm si IK et FK sont cohérentes).

Lancer dans VS Code :
    pip install matplotlib
    python sim_view.py              # fenêtre animée
    python sim_view.py --save-only  # pas de fenêtre : écrit sim_result.png et sim_result.gif

Aucune connexion au robot n'est faite : arm_motion.connect(dry_run=True).
"""
import sys

import matplotlib

if "--save-only" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import arm_motion as am

# ----------------------------------------------------------------------------
# CONFIGURATION DU TEST
# ----------------------------------------------------------------------------
ORIGIN_X_CM = 10.0      # distance (cm) pied du bras -> origine, dans l'axe ; None = origine optimale calculée
SIM_Z_CM = 10.0        # hauteur de la pince pour la simulation (cm)
GRID_CM = 15           # le quadrillage va de -15 à +15 cm
FRAMES_PER_MOVE = 25

# Positions de test (dx_cm, dy_cm) relatives à l'origine ; +dx = avant, +dy = gauche
TESTS = [
    (0, 0),
    (5, 0),
    (0, 0),
    (0, 5),
    (0, 0),
]


SHORT = {"shoulder_pan.pos": "pan", "shoulder_lift.pos": "lift", "elbow_flex.pos": "coude",
         "wrist_flex.pos": "poignet", "wrist_roll.pos": "roll"}


def tip_rel_cm(joints, x0, y0):
    """Position du bout de pince (cm) relative à l'origine, via cinématique directe."""
    x, y, _ = am.fk_xy(joints)
    return am.AXIS_X_SIGN * (x - x0) * 100.0, am.AXIS_Y_SIGN * (y - y0) * 100.0


def reachable_map(x0, y0, z0):
    """Grille 1 cm : inclinaison de pince (deg) si atteignable, NaN sinon."""
    vals = np.arange(-GRID_CM, GRID_CM + 1, 1.0)
    img = np.full((len(vals), len(vals)), np.nan)
    for i, dx in enumerate(vals):
        for j, dy in enumerate(vals):
            try:
                _, tilt = am.ik_xy(x0 + am.AXIS_X_SIGN * am._mm(dx),
                                   y0 + am.AXIS_Y_SIGN * am._mm(dy), z0, return_tilt=True)
                img[i, j] = tilt
            except ValueError:
                pass
    return vals, img


ORIGIN_X_CM_USED = 0.0


def simulate():
    """Exécute TESTS en dry run, renvoie la liste des frames et le résumé."""
    if ORIGIN_X_CM is None:
        am.use_best_origin(SIM_Z_CM)
        global ORIGIN_X_CM_USED
        ORIGIN_X_CM_USED = am.ORIGIN_OVERRIDE[0] * 100
    else:
        am.ORIGIN_OVERRIDE = (ORIGIN_X_CM / 100.0, 0.0, SIM_Z_CM / 100.0)
        ORIGIN_X_CM_USED = ORIGIN_X_CM
    am.connect(dry_run=True)
    x0, y0, z0 = am._origin
    # départ du bras = origine (le bras est supposé posé au repos au-dessus du centre)
    try:
        start = am.ik_xy(x0, y0, z0)
    except ValueError as e:
        print(f"\nL'origine elle-même (0,0) est inatteignable : {e}")
        print("-> change ORIGIN_X_CM / SIM_Z_CM, ou lance am.use_best_origin() pour une origine atteignable.")
        sys.exit(1)
    am._state = dict(start, **{"gripper.pos": am.GRIPPER_CLOSED})

    frames, summary = [], []
    prev = dict(start)
    for k, (dx, dy) in enumerate(TESTS):
        label = f"test {k + 1}/{len(TESTS)} : dx={dx:+.1f} cm, dy={dy:+.1f} cm"
        try:
            target = am.move_xy(dx, dy)
            joints = {key: target[key] for key in prev}
            for f in range(1, FRAMES_PER_MOVE + 1):
                a = f / FRAMES_PER_MOVE
                a = a * a * (3 - 2 * a)
                q = {key: prev[key] + (joints[key] - prev[key]) * a for key in prev}
                px, py = tip_rel_cm(q, x0, y0)
                frames.append(dict(px=px, py=py, label=label, joints=q, target=(dx, dy), ok=True,
                                   done=(f == FRAMES_PER_MOVE)))
            fx, fy = tip_rel_cm(joints, x0, y0)
            err_mm = 10.0 * np.hypot(fx - am._mm(dx) * 100, fy - am._mm(dy) * 100)
            _, tilt = am.ik_xy(x0 + am.AXIS_X_SIGN * am._mm(dx), y0 + am.AXIS_Y_SIGN * am._mm(dy), z0,
                               return_tilt=True)
            summary.append((dx, dy, "ATTEINT", tilt, err_mm))
            prev = joints
        except ValueError as e:
            px, py = tip_rel_cm(prev, x0, y0)
            for _ in range(FRAMES_PER_MOVE // 2):
                frames.append(dict(px=px, py=py, label=label + "  -> REFUSE", joints=prev,
                                   target=(dx, dy), ok=False, done=False))
            frames[-1]["done"] = True
            summary.append((dx, dy, "REFUSE", None, None))
            print("   raison :", e)
    return frames, summary, (x0, y0, z0)


def print_summary(summary):
    print("\n=== Résumé ===")
    print(f"{'dx cm':>7} {'dy cm':>7}  {'statut':<8} {'pince (deg)':>11} {'erreur IK<->FK':>15}")
    for dx, dy, st, tilt, err in summary:
        t = f"{tilt:.0f}" if tilt is not None else "-"
        e = f"{err:.3f} mm" if err is not None else "-"
        print(f"{dx:7.1f} {dy:7.1f}  {st:<8} {t:>11} {e:>15}")
    errs = [s[4] for s in summary if s[4] is not None]
    if errs:
        verdict = "OK" if max(errs) < 0.5 else "INCOHERENT : verifier ik/fk"
        print(f"Erreur max IK<->FK : {max(errs):.3f} mm -> {verdict}")
    print("Rappel : ceci valide la cohérence du CODE, pas la calibration du vrai bras.")


def draw(frames, summary, origin):
    x0, y0, z0 = origin
    vals, img = reachable_map(x0, y0, z0)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ext = [vals[0] - 0.5, vals[-1] + 0.5]
    # axes : horizontal = dy (gauche à gauche, donc axe inversé), vertical = dx (avant en haut)
    ax.imshow(img, origin="lower", extent=[ext[1], ext[0], ext[0], ext[1]], cmap="Greens_r",
              vmin=0, vmax=am.MAX_TILT_DEG * 2, alpha=0.55, aspect="equal")
    ax.set_xlim(GRID_CM + 0.5, -GRID_CM - 0.5)
    ax.set_ylim(-GRID_CM - 0.5, GRID_CM + 0.5)
    ax.set_xticks(range(-GRID_CM, GRID_CM + 1, 5))
    ax.set_yticks(range(-GRID_CM, GRID_CM + 1, 5))
    ax.set_xticks(np.arange(-GRID_CM - 0.5, GRID_CM + 1.5, 1), minor=True)
    ax.set_yticks(np.arange(-GRID_CM - 0.5, GRID_CM + 1.5, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", lw=0.4)
    ax.grid(which="major", color="#777777", lw=0.8)
    ax.set_xlabel(f"dy (cm)  <- gauche | droite ->\n(base du robot : {ORIGIN_X_CM_USED:.1f} cm derrière l'origine, hors cadre en bas)")
    ax.set_ylabel("dx (cm)  arrière | avant (s'éloigne de la base)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.plot(0, 0, marker="s", color="k", ms=7, label="origine (repos)")

    # cibles : marquées au fur et à mesure
    for dx, dy, st, _, _ in summary:
        pass
    trail, = ax.plot([], [], "-", color="#1f77b4", lw=1.5, label="trajet du bout de pince")
    tip, = ax.plot([], [], "o", color="#d62728", ms=12, mec="k", label="bout de pince")
    ok_pts, = ax.plot([], [], "x", color="green", ms=11, mew=3, label="cible atteinte")
    ko_pts, = ax.plot([], [], "x", color="red", ms=11, mew=3, label="cible REFUSEE")
    info = ax.text(0.01, 1.02, "", transform=ax.transAxes, fontsize=9, va="bottom", family="monospace")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Simulation (aucun robot) - vert foncé = pince verticale, clair = inclinée (max 20°)", fontsize=10, pad=34)

    hist = dict(x=[], y=[], okx=[], oky=[], kox=[], koy=[])

    def update(i):
        f = frames[i]
        hist["x"].append(f["py"])
        hist["y"].append(f["px"])
        trail.set_data(hist["x"], hist["y"])
        tip.set_data([f["py"]], [f["px"]])
        if f["done"]:
            tdx, tdy = f["target"]
            key = "ok" if f["ok"] else "ko"
            hist[key + "x"].append(tdy)
            hist[key + "y"].append(tdx)
            ok_pts.set_data(hist["okx"], hist["oky"])
            ko_pts.set_data(hist["kox"], hist["koy"])
        q = f["joints"]
        info.set_text(f["label"] + "\n" + " ".join(f"{SHORT[k]}={v:6.1f}" for k, v in q.items()))
        return trail, tip, ok_pts, ko_pts, info

    anim = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False, repeat=False)
    return fig, anim, update


def main():
    frames, summary, origin = simulate()
    print_summary(summary)
    fig, anim, update = draw(frames, summary, origin)
    if "--save-only" in sys.argv:
        for i in range(len(frames)):
            update(i)
        fig.savefig("sim_result.png", dpi=130)
        print("Écrit : sim_result.png")
        anim.save("sim_result.gif", writer=PillowWriter(fps=25))
        print("Écrit : sim_result.gif")
    else:
        plt.show()


if __name__ == "__main__":
    main()