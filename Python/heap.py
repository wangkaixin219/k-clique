import numpy as np


def parent(i):
    assert i >= 1
    return (i - 1) >> 1


def left_child(i):
    return (i << 1) + 1


def right_child(i):
    return (i + 1) << 1


class MaxHeap(object):
    def __init__(self, n):
        self.n = n
        self.heap = [None] * self.n            # store a list of (key, value) pairs, key is node idx, value is deg
        self.pos = [-1] * self.n             # the index of the node in heap
        self.cur_size = 0

    def max_elem(self):
        return self.heap[0]

    def top_k(self, k):
        res = []
        while k > 0:
            elem = self.max_elem()
            res.append(elem)
            self.pop()
            k -= 1
        return res

    def insert(self, elem):
        i = self.cur_size
        self.heap[i] = elem           # elem = (key, value) tuple
        k, v = elem
        self.pos[k] = i
        self.cur_size += 1
        self._bubble_up(i)

    def pop(self):
        self.pos[self.heap[0][0]] = -1
        self.cur_size -= 1
        self.heap[0] = self.heap[self.cur_size]
        self.pos[self.heap[0][0]] = 0
        self._bubble_down(0)

    def clear(self):
        self.heap = [None] * self.n
        self.pos = [-1] * self.n
        self.cur_size = 0

    def empty(self):
        return self.cur_size == 0

    def _bubble_up(self, i):
        while i > 0 and self.heap[i][1] > self.heap[parent(i)][1]:
            self._swap(i, parent(i))
            i = parent(i)

    def _bubble_down(self, i):
        l = left_child(i)
        r = right_child(i)

        while l < self.cur_size:
            child = r if r < self.cur_size and self.heap[r][1] > self.heap[l][1] else l
            if self.heap[i][1] < self.heap[child][1]:
                self._swap(i, child)
                i = child
                l = left_child(i)
                r = right_child(i)
                continue
            break

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos[self.heap[i][0]] = i
        self.pos[self.heap[j][0]] = j


