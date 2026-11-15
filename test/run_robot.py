
import arm_motion as am

# Coordonnées en cm par rapport à l'origine calibrée
POINTS = [
    (0.0, 0.0),
    (5.0, 0.0),   # Point 1
    (5.0, 5.0),   # Point 2
    (0.0, 0.0),   # Retour à l'origine
]

# Hauteur de déplacement au-dessus de la table
Z_CM = 3.0


def main():
    connected = False

    try:
        print("Connexion au SO-101...")
        am.connect(dry_run=False)
        connected = True

        print("Connexion réussie.")
        print("Position articulaire actuelle :", am.read_joints())

        input(
            "\nVérifie que la zone est dégagée. "
            "Appuie sur Entrée pour commencer, Ctrl+C pour annuler."
        )

        print("\nDéplacement vers l'origine calibrée...")
        am.move_xy(0.0, 0.0, z_cm=Z_CM)

        for i, (x, y) in enumerate(POINTS[:-1], start=1):
            input(f"\nPrêt à aller au point {i} ({x}, {y}) cm ? Entrée pour continuer...")
            am.move_xy(x, y, z_cm=Z_CM)

        input("\nPrêt à revenir à l'origine ? Entrée pour continuer...")
        am.move_xy(0.0, 0.0, z_cm=Z_CM)

        print("\nParcours terminé.")

    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur.")

    except Exception as e:
        print(f"\nErreur pendant l'exécution : {e}")
        raise

    finally:
        if connected:
            am.disconnect()
            print("Bras déconnecté.")


if __name__ == "__main__":
    main()
