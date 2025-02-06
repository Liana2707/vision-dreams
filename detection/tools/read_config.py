import argparse
import json
import os
import sys
import pandas as pd

"""
Read configuration parameters
from file, command line arguments or environment variables
"""

def environ_get():
    args = pd.Series()
    args['input_image'] = os.environ.get('INPUT_IMAGE')
    args['method'] = os.environ.get('METHOD', 'yolo')
    args['output'] = os.getenv("OUTPUT", '.')

    args['rabbit'] = bool(os.environ.get('RABBIT', 'True').lower() in ('true', '1', 't'))
    args['output_host'] = os.environ.get('OUTPUT_HOST', 'amqp://admin:Xu4lCyXQnOpfINud6QLYo@172.29.133.4:7672/vhost1')
    args['output_queue'] = os.environ.get('OUTPUT_QUEUE', 'det_to_track')
    return args


def arg_parser():
    parser = argparse.ArgumentParser(description='Detection module')
    parser.add_argument('-f', '--file', metavar='Config file', type=str,
                        help='Path to Config file')
    parser.add_argument('-i', '--input_image', metavar='Input',
                        type=str,  help='Path to image source')
    parser.add_argument('-b', '--input_bbox',
                        metavar='Input_bbox', type=str, help='Path to file with bbox')
    parser.add_argument('-fb', '--format_bbox', default='yolo', metavar='Format_bbox', type=str,
                        help='Format bbox', choices=['pascal_voc', 'albumentations', 'coco',  'yolo'])
    parser.add_argument('-m', '--method', default='yolo', metavar='Method', type=str,
                        help='One of available tracking algorithms: yolo',
                        choices=['yolo', 'ssdlite', 'yolos'])
    parser.add_argument('-o', '--output', default='.', metavar='Output',
                        type=str, help='Path to output data')
    parser.add_argument('-sj', '--save_json', action='store_true', help='Save in json format')
    parser.add_argument('-s', '--save_image', action='store_true', help='Save in image format')
    parser.add_argument('-v', '--visualization', action='store_true', help='Show visualization')
    parser.add_argument('-r', '--rabbit', action='store_false', help='Send bbox w RabbitMQ')
    parser.add_argument('-oh', '--output_host', type=str, help='RabbitMQ host')
    parser.add_argument('-oq', '--output_queue', type=str, help='RabbitMQ queue')
    args = parser.parse_args()
    return args


def read_input():
    args = arg_parser()
    if args.file:
        with open(args.file, 'r') as file:
            df = json.load(file)
        args = pd.Series(df)
    elif args.input_image:
        args = arg_parser()
    elif os.environ.get('INPUT_IMAGE') is not None:
        args = environ_get()
    else:
        print("Could not read input")
        sys.exit()
    return args





