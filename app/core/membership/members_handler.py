# Saber que meinbros estan vivos

# Logica para manterner haciendolo cada cierto tiempo

from config.manager import MEMBERSHIPS_TIME_REFRESH
from typing import List
from threading import Thread
import time


class MemberShips:
    def __init__(self):
        self.members: List[str] = []

    def refresh_members(self):
        pass  # < ========== aqui falta la logica para la busqueda de los nodos conectados

    def th_get_members(self):
        while True:
            self.refresh_members()
            time.sleep(MEMBERSHIPS_TIME_REFRESH)

    def start(self):
        th = Thread(self.th_get_members)
        th.start()
