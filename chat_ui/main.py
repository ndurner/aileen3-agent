from chat_ui.ui.app import build_app


def main() -> None:
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
