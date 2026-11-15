
import arm_motion as am

# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

# Deux points relatifs à l'origine (en cm)
POINTS = [
    (5.0, 0.0),
    (5.0, 5.0),
]

# Hauteur de la pince au-dessus de la table
Z_CM = 10.0

def main():

    # Connexion sans robot
    am.connect(dry_run=True)

    try:
        print("\n1. Départ vers l'origine (0,0)")
        am.move_xy(0, 0, z_cm=Z_CM)

        input("Appuyer sur Entrée pour commencer...")

        # Aller successivement aux deux points
        for i, (x, y) in enumerate(POINTS):

            print(f"\n2. Déplacement vers le point {i+1}")
            print(f"   X = {x} cm, Y = {y} cm")

            am.move_xy(x, y, z_cm=Z_CM)

            input("Appuyer sur Entrée pour continuer...")

        # Retour à l'origine
        print("\n3. Retour à l'origine (0,0)")

        am.move_xy(0, 0, z_cm=Z_CM)

        print("\nTrajectoire terminée !")

    except ValueError as e:
        print("Déplacement impossible :", e)

    finally:
        am.disconnect()


if __name__ == "__main__":
    main()
