
import argparse
from lm_definitions import LMs

lmdefs = list(LMs.values())

# TODO: Implement run_experiments. Because of limited space we have to do one LM at a time:
#  store results from one LM
#  delete that LM 
#  load the next LM
#  then compare all the results
