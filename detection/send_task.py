from publishers.rabbitmqpublisher import RabbitMQPublisher

def interactive_shell(queue, host, command, mainLogger ):
    publisher = RabbitMQPublisher(host=host, mainLogger = mainLogger)
    publisher.publish(topic=queue, message=command)

