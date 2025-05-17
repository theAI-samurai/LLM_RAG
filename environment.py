import os

from dotenv import load_dotenv

load_dotenv()

PIPELINE_CONFIG = "./config/pipeline_config.json"

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
LOG_LEVEL = os.getenv("LOG_LEVEL", "debug")

AUTH_TYPE = os.getenv("OCI_AUTH_TYPE", "API_KEY")
OCI_PROFILE = os.getenv("OCI_PROFILE", "DEFAULT")
VAULT_ID = os.getenv("VAULT_ID", None)
REGION = os.getenv("REGION", "us-chicago-1")
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
ENVIRONMENT = os.getenv("ENVIRONMENT")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

PORTFOLIO_JSON = "./config/portfolios.json"
APP_CONFIG = "./config/app_config.json"

ORACLE_DB_HOST = os.getenv("ORACLE_DB_HOST")
ORACLE_DB_DSN = os.getenv("ORACLE_DB_DSN")
ORACLE_DB_USER = os.getenv("ORACLE_DB_USER")
ORACLE_DB_PASSWORD = os.getenv("ORACLE_DB_PASSWORD")
ORACLE_DB_USER_SECRET_ID = os.getenv("ORACLE_DB_USER_SECRET_ID")
ORACLE_DB_PASSWORD_SECRET_ID = os.getenv("ORACLE_DB_PASSWORD_SECRET_ID")
ORACLE_DB_CONNECTION_STRING = os.getenv("ORACLE_DB_CONNECTION_STRING")

FRONT_END_URL = os.getenv("FRONT_END_URL", "http://localhost:5173")
TRANSLATOR_TO_USE = os.getenv(
    "TRANSLATOR_TO_USE", "LLM_TRANSLATOR"
)  # Allowed values: LLM_TRANSLATOR, OCIAI_TRANSLATOR

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_USE_TLS = os.getenv("REDIS_USE_TLS", "false").lower() == "true"

GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")

PROMPT_TEMPLATES_BUCKET = os.getenv("PROMPT_TEMPLATES_BUCKET")
DATA_BUCKET = os.getenv("DATA_BUCKET")

ENABLE_PROMPT_LOGGING= os.getenv("ENABLE_PROMPT_LOGGING", "false").lower() == 'true'

MODEL_ID = os.getenv("MODEL_ID","cohere.command-r-plus-08-2024")
MODEL_ID_CR = os.getenv("MODEL_ID_CR","cohere.command-r-08-2024")
