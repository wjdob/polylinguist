from __future__ import annotations

import argparse

from polylinguist.config import AppConfig, AppPaths
from polylinguist.services.runtime import create_services


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove Polylinguist local settings, caches, and Polylinguist-managed model artifacts."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    paths = AppPaths.detect()
    config = AppConfig.detect()

    if not args.yes:
        print(f"Polylinguist local state directory: {paths.root}")
        print("This removes Polylinguist settings, generated subtitle cache, and Polylinguist-managed model artifacts.")
        print("Shared Hugging Face and Argos caches are left in place.")
        response = input("Continue with local uninstall? [y/N]: ").strip().lower()
        if response not in {"y", "yes"}:
            print("Cancelled.")
            return

    services = create_services(paths, config)
    result = services.uninstall_local_data()

    print(result["detail"])
    if result["removed_paths"]:
        print("\nRemoved paths:")
        for item in result["removed_paths"]:
            print(f"- {item}")
    if result["notes"]:
        print("\nNotes:")
        for item in result["notes"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
