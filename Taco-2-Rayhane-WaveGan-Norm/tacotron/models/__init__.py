from .tacotron import Tacotron


def create_model(name, hparams):
  if name == 'Tacotron':
    return Tacotron(hparams)
  if name == 'Tacotron2-mix-phoneme':
    return Tacotron(hparams)
  
  raise Exception('Unknown model: ' + name)
