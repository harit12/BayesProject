from movieviewerModel import Model
from movieviewerView import View
class Controller:
    def __init__(self,model, view):
       self.model = model
       self.view = view
    def sentenceGrabber(self):
        view = self.view
        view.initial()
        answer = input()
        return answer
    def getResults(self, answer):
        view = self.view
        model = self.model
        posProb, negProb, results = model.finalProb(answer)
        view.displayResults(posProb, negProb, results)
def main():
    model = Model()
    view = View()
    control = Controller(model, view)
    answer = control.sentenceGrabber()
    control.getResults(answer)
if __name__ == "__main__":
    main()
