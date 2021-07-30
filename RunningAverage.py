class RunningAverage():
  '''
  A simple class which maintains the running average of loss or accuracy stats
  '''
  def __init__(self):
    self.accum = 0.0
    self.counts = 0

  def update(self, val, batch_size):
    self.accum += val
    self.counts += batch_size

  def __call__(self):
    return self.accum/self.counts if self.counts != 0 else None