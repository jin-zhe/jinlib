class RunningAverage():
  '''
  A simple class which maintains the running average of loss or accuracy stats
  '''
  def __init__(self):
    self.accum = 0.0
    self.counts = 0

  def update(self, bag_sum, bag_size=1):
    self.accum += bag_sum
    self.counts += bag_size

  def __call__(self):
    return self.accum/self.counts if self.counts != 0 else None