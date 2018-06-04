from config import get_args
from model import hier_rnn

def main(args):
    network=hier_rnn(args)
    network.train()
if __name__=='__main__':
    args=get_args()
    main(args)