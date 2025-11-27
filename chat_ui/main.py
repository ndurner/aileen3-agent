from env_support import ensure_env_loaded
from chat_ui.ui.app import build_app


def main() -> None:
    ensure_env_loaded()
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
