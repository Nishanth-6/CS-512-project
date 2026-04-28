class node:
    def __init__(self, value):
        self.value = value
        self.next = None

class makeAndTraverse:
    def __init__(self):
        self.head = None

    def add(self, value):
        new_node = node(value)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def traverse(self):
        current_node = self.head
        while current_node:
            print(current_node.value)
            current_node = current_node.next

if __name__ == "__main__":
    linked_list = makeAndTraverse()
    linked_list.add(1)
    linked_list.add(2)
    linked_list.add(21)

    print("Traversing the linked list:")
    linked_list.traverse()