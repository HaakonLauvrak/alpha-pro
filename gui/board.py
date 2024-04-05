from abc import ABC, abstractmethod

class BOARD(ABC):

    @abstractmethod
    def get_state(): 
        pass
    
    @abstractmethod
    def get_ann_input():
        pass

    