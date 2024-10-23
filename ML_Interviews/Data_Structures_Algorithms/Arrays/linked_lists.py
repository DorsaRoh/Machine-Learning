"""
SOLUTIONS
"""



"""
Reverse Linked List
https://leetcode.com/problems/reverse-linked-list/ 
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        past = None
        curr = head
        
        while curr != None:
            next_node = curr.next
            curr.next = past

            past = curr
            curr = next_node
        
        return past
        