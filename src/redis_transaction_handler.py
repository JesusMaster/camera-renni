import redis
import datetime
import json

class RedisTransactionHandler:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.redis_client = redis.Redis.from_url(redis_url)
        try:
            self.redis_client.ping()
            print("Connected to Redis successfully!")
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")

    def save_transaction(self, transaction_type, actions):
        now = datetime.datetime.now()
        key = now.strftime("%d:%m:%Y:%H:%M:") + transaction_type
        try:
            for action in actions:
                self.redis_client.rpush(key, action)
            #print(f"Transaction actions saved to Redis with key: {key}")
        except redis.exceptions.RedisError as e:
            print(f"Error saving transaction actions to Redis: {e}")

    def save_log(self, log_data):
        now = datetime.datetime.now()
        key = now.strftime("%d:%m:%Y") + ":logs"
        try:
            self.redis_client.rpush(key, json.dumps(log_data))
            #print(f"Transaction log saved to Redis with key: {key}")
        except redis.exceptions.RedisError as e:
            print(f"Error saving transaction log to Redis: {e}")
