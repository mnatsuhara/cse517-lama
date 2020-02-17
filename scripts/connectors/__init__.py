# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .bert_connector import Bert
from .elmo_connector import Elmo
from .transformerxl_connector import TransformerXL

def build_model_by_name(lm, args, verbose=True):
    """Load a model identified by the "lm" parameter (not by args.lm), passing args to the connector.
    """
    MODEL_NAME_TO_CLASS = dict(
        elmo=Elmo,
        bert=Bert,
        transformerxl=TransformerXL,
    )
    initializer = MODEL_NAME_TO_CLASS.get(lm)
    if initializer is None:
        raise ValueError("Unrecognized Language Model: %s." % lm)
    if verbose:
        print("Loading %s model..." % lm)
    return MODEL_NAME_TO_CLASS[lm](args)
