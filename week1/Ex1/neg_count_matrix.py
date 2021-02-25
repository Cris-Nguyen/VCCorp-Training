from typing import List


class Solution:

    def count_neg_matrix_ver1(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])

        # Initialize i, j and count element
        i, j = 0, n - 1
        count = 0

        while i < m:
            # Loop until the first element in that row < 0
            while j >= 0 and matrix[i][j] >= 0:
                j -= 1
            count += (j + 1)
            i += 1

        return count

    def count_neg_matrix_ver2(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])

        # Initialize count element
        count = 0

        for i in range(m):
            # Binary search
            left, right = 0, n - 1
            while left <= right:
                mid = (left + right) // 2
                if matrix[i][mid] >= 0:
                    right = mid - 1
                else:
                    left = mid + 1
            count += left

        return count


if __name__ == '__main__':
    solution = Solution()
    test_cases = [{'matrix': [[-1]], 'target': 1}, {'matrix': [[-5, -2], [-1, -3]], 'target': 4}, {'matrix': [[1, 3], [2, 4]], 'target': 0}, {'matrix': [[-5, -2, 3], [-2, 1, 4], [0, 4, 9]], 'target': 3}, {'matrix': [[-9, -7, -5, -4], [-8, -6, -4, -2], [-5, -3, 1, 2], [-4, 1, 2, 3]], 'target': 11}]
    for i in range(len(test_cases)):
        assert(solution.count_neg_matrix_ver1(test_cases[i]['matrix']) == test_cases[i]['target'])
        assert(solution.count_neg_matrix_ver2(test_cases[i]['matrix']) == test_cases[i]['target'])
