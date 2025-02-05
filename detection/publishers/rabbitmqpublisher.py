from typing import Any
import pika
from publishers.publisher import Publisher


# RabbitMQPublisher
class RabbitMQPublisher(Publisher):
    def __init__(self, host: str, mainLogger):
        self.connection = pika.BlockingConnection(pika.URLParameters(host))
        self.channel = self.connection.channel()
        self.mainLogger = mainLogger

    def publish(self, topic: str, message: Any) -> None:
        self.channel.queue_declare(queue=topic)
        self.channel.basic_publish(exchange='', routing_key=topic, body=str(message))
        self.mainLogger.info(f"[RabbitMQ] Published to {topic}: {message}")

