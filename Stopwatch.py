import datetime

class Stopwatch():
  def __init__(self):
    self.reset()

  def reset(self):
    self.start = datetime.datetime.now()

  def __call__(self):
    elapsed = datetime.datetime.now() - self.start
    return str(elapsed)