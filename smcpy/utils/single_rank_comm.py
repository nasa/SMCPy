class SingleRankComm:
    def __init__(self):
        pass

    @staticmethod
    def Get_rank():
        return 0

    @staticmethod
    def Get_size():
        return 1

    @staticmethod
    def scatter(scatter_list, *args, **kwargs):
        if len(scatter_list) == 1:
            return scatter_list[0]
        else:
            raise ValueError(
                "SingleRankComm cannot accept more than one " + "partition"
            )

    @staticmethod
    def allgather(scatter_list, *args, **kwargs):
        return [scatter_list]
