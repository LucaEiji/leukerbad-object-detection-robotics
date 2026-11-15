"""Shared opt-in guard for project scripts that access physical devices."""

from __future__ import annotations

import argparse


def add_hardware_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Autoriser explicitement l'accès au matériel réel (désactivé par défaut).",
    )


def require_hardware(parser: argparse.ArgumentParser, enabled: bool) -> None:
    if not enabled:
        parser.error("Accès matériel désactivé. Ajouter explicitement --hardware après vérification et supervision sur place.")
