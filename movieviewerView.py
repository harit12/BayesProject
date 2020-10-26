class View:
    def initial(self):
        print("Type review:")
    def displayResults(self, pos, neg, res):
        print("The probability of review being positive is: ", pos)
        print("The probability of review being negative is: ", neg)
        print("Therefore, the review is ", res)
