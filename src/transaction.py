from time import time

class Transaction:
    def __init__(self, client_id, cashier_id, config):
        self.start_time = time()
        self.client_id = client_id
        self.cashier_id = cashier_id
        self.config = config
        self.events = []
        self.event_set = set()

    def add_event(self, event_name):
        if event_name not in self.event_set:
            self.events.append(event_name)
            self.event_set.add(event_name)

    def get_events(self):
        return self.events

    def get_duration(self):
        return time() - self.start_time
