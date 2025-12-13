from typing import List
from abc import ABC, abstractclassmethod


class BuildMethod(ABC):

    def __init__(self, target : str, build_method_type : str, nvp : List):

        self.target = target
        self.bm_type = build_method_type
        self.nvp = nvp
    

class BuildMethodColleciton:

    # dict of ubild method : (target , type) => BuildMethod Object

    pass
