from .binary_linear import BinaryLinear
from .mlc_linear import MLCLinear
from .chief import CHIEF
from .ml_decoder import MLDecoder
from .query2label import Query2Label
from .wscer_mlc import WSCerMLC

allowed_classifier_type = ['binary_linear', 'mlc_linear', 'chief', 'ml_decoder', 'query2label', 'wscer_mlc']

def get_classifier(args):
    classifier_type = args.classifier_type
    assert classifier_type in allowed_classifier_type, f'classifier_type allowed in {allowed_classifier_type}'
    
    classifier = None
    if classifier_type == 'binary_linear':
        classifier = BinaryLinear
    if classifier_type == 'mlc_linear':
        classifier = MLCLinear
    if classifier_type == 'chief':
        classifier = CHIEF
    if classifier_type == 'ml_decoder':
        classifier = MLDecoder
    if classifier_type == 'query2label':
        classifier = Query2Label
    if classifier_type == 'wscer_mlc':
        classifier = WSCerMLC
    
    return classifier(args)