import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MOCK_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_db.json")

class DatabaseClient:
    def __init__(self):
        self.client = None
        self.db = None
        self.is_mock = False
        
        try:
            # Attempt to connect to MongoDB (2 second timeout)
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            # The ping command is cheap and will force a connection test
            self.client.admin.command('ping')
            self.db = self.client["customer_churn_db"]
            logger.info("Successfully connected to MongoDB server.")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"Could not connect to MongoDB: {e}. Falling back to JSON Mock Database at {MOCK_DB_PATH}")
            self.is_mock = True
            self._init_mock_db()

    def _init_mock_db(self):
        """Initializes the mock database file if it doesn't exist."""
        if not os.path.exists(MOCK_DB_PATH):
            with open(MOCK_DB_PATH, "w") as f:
                json.dump({"predictions": [], "retention_plans": []}, f, indent=4)
            logger.info("Initialized mock JSON database.")

    def _read_mock_db(self):
        """Reads data from the mock database file."""
        if not os.path.exists(MOCK_DB_PATH):
            self._init_mock_db()
        try:
            with open(MOCK_DB_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading mock database: {e}")
            return {"predictions": [], "retention_plans": []}

    def _write_mock_db(self, data):
        """Writes data to the mock database file."""
        try:
            with open(MOCK_DB_PATH, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Error writing to mock database: {e}")

    def save_prediction(self, customer_data, risk_score, risk_level):
        """
        Saves a prediction run to the database.
        """
        record = {
            "customer_id": customer_data.get("customerID", "TEMP_" + datetime.now().strftime("%Y%m%d%H%M%S")),
            "timestamp": datetime.now().isoformat(),
            "customer_features": customer_data,
            "risk_score": float(risk_score),
            "risk_level": risk_level
        }
        
        if not self.is_mock:
            try:
                result = self.db.predictions.insert_one(record)
                logger.info(f"Saved prediction to MongoDB. Inserted ID: {result.inserted_id}")
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Failed to save prediction to MongoDB: {e}. Attempting mock save.")
                # Fall through to mock save
        
        # Mock database save
        db_data = self._read_mock_db()
        # Generate mock string ID
        record["_id"] = "mock_pred_" + str(len(db_data["predictions"]) + 1)
        db_data["predictions"].append(record)
        self._write_mock_db(db_data)
        logger.info(f"Saved prediction to mock database. ID: {record['_id']}")
        return record["_id"]

    def save_retention_plan(self, customer_id, risk_score, reasons, strategies, offers, email_subject, email_body):
        """
        Saves the AI-generated retention plan for a customer.
        """
        record = {
            "customer_id": customer_id,
            "timestamp": datetime.now().isoformat(),
            "risk_score": float(risk_score),
            "reasons": reasons,
            "strategies": strategies,
            "offers": offers,
            "email_subject": email_subject,
            "email_body": email_body,
            "status": "Generated" # Can be updated to "Sent" or "Accepted"
        }
        
        if not self.is_mock:
            try:
                result = self.db.retention_plans.insert_one(record)
                logger.info(f"Saved retention plan to MongoDB. Inserted ID: {result.inserted_id}")
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Failed to save retention plan to MongoDB: {e}. Attempting mock save.")
        
        # Mock database save
        db_data = self._read_mock_db()
        record["_id"] = "mock_plan_" + str(len(db_data["retention_plans"]) + 1)
        db_data["retention_plans"].append(record)
        self._write_mock_db(db_data)
        logger.info(f"Saved retention plan to mock database. ID: {record['_id']}")
        return record["_id"]

    def get_prediction_history(self, limit=100):
        """
        Retrieves prediction history sorted by timestamp descending.
        """
        if not self.is_mock:
            try:
                cursor = self.db.predictions.find().sort("timestamp", -1).limit(limit)
                history = []
                for item in cursor:
                    item["_id"] = str(item["_id"])
                    history.append(item)
                return history
            except Exception as e:
                logger.error(f"Failed to fetch predictions from MongoDB: {e}")
        
        # Mock DB fallback
        db_data = self._read_mock_db()
        history = db_data["predictions"].copy()
        # Sort by timestamp descending
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history[:limit]

    def get_retention_plans(self, limit=100):
        """
        Retrieves retention plan history sorted by timestamp descending.
        """
        if not self.is_mock:
            try:
                cursor = self.db.retention_plans.find().sort("timestamp", -1).limit(limit)
                plans = []
                for item in cursor:
                    item["_id"] = str(item["_id"])
                    plans.append(item)
                return plans
            except Exception as e:
                logger.error(f"Failed to fetch retention plans from MongoDB: {e}")
        
        # Mock DB fallback
        db_data = self._read_mock_db()
        plans = db_data["retention_plans"].copy()
        plans.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return plans[:limit]

# Singleton instance
db_client = DatabaseClient()
