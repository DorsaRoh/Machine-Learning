"""
SOLUTIONS
"""


"""
Valid Parenthesis: 
https://leetcode.com/problems/valid-parentheses/description/
"""

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []

        bracket_rep = {'(': 1, '[': 2, '{': 3, ')':-1, ']':-2,'}':-3}

        if len(s) < 2:
            return False

        for i in range(len(s)):
            if bracket_rep[s[i]] > 0:   # open
                stack.append(s[i])
            elif bracket_rep[s[i]] < 0: # closed
                if len(stack) >= 1 and bracket_rep[s[i]] + bracket_rep[stack[-1]] == 0: # cancelled
                    stack.pop()
                else:
                    return False

        return len(stack) == 0


"""
Min Stack:
https://leetcode.com/problems/min-stack/description/
"""

class MinStack:

    def __init__(self):
        self.stack = []
        self.min = [sys.maxsize - 1]

    def push(self, val: int) -> None:
        # adding an element to the "top" of a stack
        self.stack.append(val)
        if val <= self.min[-1]:
            self.min.append(val)
        
    def pop(self) -> None:
        # removing the topmost element
        if self.stack[-1] == self.min[-1]:
            self.min.pop()

        self.stack.pop() 

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min[-1]
        

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()