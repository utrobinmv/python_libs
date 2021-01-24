import pika

def rabbit_mq_connection_connections(rmq_url_connection_str, user, password):

    credentials = pika.PlainCredentials(user, password)

    rmq_url_connection_list = rmq_url_connection_str.split('|')

    parameters = []

    for connect_str in rmq_url_connection_list:
        connect_list = connect_str.split(':')
        if len(connect_list) > 1:
            port = connect_list[1]
        else:
            port = 5672
        host = connect_list[0]

        parameters.append(pika.ConnectionParameters(host = host, port = port, credentials = credentials))


    # rmq_parameters = pika.URLParameters(rmq_url_connection_str)
    # rmq_connection = pika.BlockingConnection(rmq_parameters)
    # rmq_channel = rmq_connection.channel()

    connection = pika.BlockingConnection(parameters)
    return connection

def rabbit_mq_connection(server_mq_host, server_mq_port):
    # rmq_parameters = pika.URLParameters(rmq_url_connection_str)
    # rmq_connection = pika.BlockingConnection(rmq_parameters)
    # rmq_channel = rmq_connection.channel()

    connection = pika.BlockingConnection(pika.ConnectionParameters(host=server_mq_host, port = server_mq_port))
    return connection

def rabbit_mq_selectconnection(server_mq_host, server_mq_port):
    connection = pika.SelectConnection(pika.ConnectionParameters(host=server_mq_host, port = server_mq_port))
    return connection

def rabbit_mq_connect_channel(connection):
    server_mq_channel = connection.channel()
    return server_mq_channel

def rabbit_mq_basic_consume(server_mq_channel, receive_queue, def_callback_request):
    server_mq_channel.queue_declare(queue=receive_queue, durable=True)
    server_mq_channel.basic_consume(queue=receive_queue, on_message_callback=def_callback_request, auto_ack=False)

def rabbit_mq_start_consuming(server_mq_channel):
    server_mq_channel.start_consuming()

def rabbit_mq_basic_publish(server_mq_channel, server_mq_send_queue, message):
    server_mq_channel.queue_declare(queue=server_mq_send_queue, durable=True)
    server_mq_channel.basic_publish(exchange='', routing_key=server_mq_send_queue, body=message)

def rabbit_mq_basic_ack(server_mq_channel, delivery_tag):
    server_mq_channel.basic_ack(delivery_tag)
