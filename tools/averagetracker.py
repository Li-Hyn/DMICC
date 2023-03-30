class AverageTracker():
    def __init__(self):
        self.step = 0
        self.cur_avg = 0

    def add(self, value):
        self.cur_avg *= self.step / (self.step + 1)
        self.cur_avg += value / (self.step + 1)
        self.step += 1

    def reset(self):
        self.step = 0
        self.cur_avg = 0

    def avg(self):
        return self.cur_avg.item()



