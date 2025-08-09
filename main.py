from src.app import AppFactory

if __name__ == "__main__":
    app = AppFactory.create_app()
    app.run()
