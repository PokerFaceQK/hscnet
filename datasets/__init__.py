from .seven_scenes import SevenScenes
from .twelve_scenes import TwelveScenes
from .cambridge_landmarks import Cambridge
from .rio10 import Rio10Dataset
from .rio10_wsz import RIOScenes

def get_dataset(name):

    return {
            '7S' : SevenScenes,
            '12S' : TwelveScenes,
            'Cambridge' : Cambridge,
            'rio10': Rio10Dataset
           }[name]
