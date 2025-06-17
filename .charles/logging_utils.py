import logging
import sys
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.WHITE,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'SUCCESS': Fore.GREEN,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, Fore.WHITE)
        
        # Color the logger name, levelname, and message
        if hasattr(record, 'name'):
            record.name = f"{log_color}{record.name}{Style.RESET_ALL}"
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{log_color}{record.msg}{Style.RESET_ALL}"
        
        return super().format(record)

def setup_logging(name=None):
    """Setup colored logging"""
    # Add custom SUCCESS level
    logging.SUCCESS = 25
    logging.addLevelName(logging.SUCCESS, 'SUCCESS')
    
    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.SUCCESS):
            self._log(logging.SUCCESS, message, args, **kwargs)
    
    logging.Logger.success = success
    
    # Configure logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    
    # Remove all handlers associated with the logger (prevents duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler with colored formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    return logger

def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = logging.getLogger(__name__)
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def setup_exception_handler():
    """Set up global exception handler"""
    sys.excepthook = handle_exception

def with_exception_logging(func):
    """Decorator to add exception logging to a function"""
    def wrapper(*args, **kwargs):
        old_hook = sys.excepthook
        sys.excepthook = handle_exception
        try:
            return func(*args, **kwargs)
        finally:
            sys.excepthook = old_hook
    return wrapper
