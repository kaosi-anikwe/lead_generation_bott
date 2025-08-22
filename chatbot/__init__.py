import logging
import os
from flask import Flask
from .config import Config, config
from logging.handlers import RotatingFileHandler


def setup_logging(app: Flask) -> logging.Logger:
    """Setup application logging with proper configuration"""
    log_level = getattr(logging, app.config.get("LOG_LEVEL", "INFO").upper())

    # Create a logger
    logger = logging.getLogger("chatbot")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # File handler with rotation
    log_filename = os.environ.get("LOG_FILE", "run.log")
    log_max_size = int(os.environ.get("LOG_MAX_SIZE", str(1 * 1024 * 1024)))  # 1 MB
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=log_max_size,
        backupCount=int(os.environ.get("LOG_BACKUP_COUNT", "5")),
    )
    file_handler.setLevel(log_level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

    console_handler.setFormatter(simple_formatter)
    file_handler.setFormatter(detailed_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = logging.getLogger("chatbot")


def create_app(config_name: str = None) -> Flask:
    """Application factory pattern"""
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "default")

    config_class = config.get(config_name, config["default"])

    # Validate configuration
    config_class.validate()

    app = Flask(__name__)
    app.config.from_object(config_class)

    # Setup logging
    global logger
    logger = setup_logging(app)

    # Register blueprints
    from .main import chatbot

    app.register_blueprint(chatbot)

    # Add health check endpoint
    @app.route("/health")
    def health_check():
        return {"status": "healthy", "service": "lead_generation_bot"}, 200

    logger.info(f"Flask app created with config: {config_name}")
    return app
