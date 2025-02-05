import re
import subprocess
import sys


def ping(host, mainLogger):

    # Регулярное выражение для поиска IP-адреса
    ip_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'

    # Поиск IP-адреса в строке
    match = re.search(ip_pattern, host)

    # Проверка и вывод результата
    if match:
        ip_address = match.group(1)
        result = subprocess.run(['ping', '-c', '1', ip_address], stdout=subprocess.PIPE)
        if result.returncode == 0:
            mainLogger.info(f'Connected to: {host}')
        else:
            mainLogger.error("Could not connect to host")
            sys.exit()
    else:
        mainLogger.error("Incorrect host")
        sys.exit()




