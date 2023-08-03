class Graph(object):
    def is_valid(self, grid, r, c):
        m, n = len(grid), len(grid[0])
        if r < 0 or c < 0 or r >= m or c >= n:
            return False
        return True

    def numIslandsDFS(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid or not grid[0]:
            return 0

        m, n = len(grid), len(grid[0])
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, r, c):
        grid[r][c] = 0
        directions = [(0,1), (0,-1), (-1,0), (1,0),(-1,-1),(1,1),(1,-1),(-1,1)]
        for d in directions:
            nr, nc = r + d[0], c + d[1]
            if self.is_valid(grid, nr, nc) and grid[nr][nc] == 1:
                self.dfs(grid, nr, nc)


    def assign_islands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid or not grid[0]:
            return 0

        m, n = len(grid), len(grid[0])
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    count += 1
                    self.dfs2(grid, i, j,count)
        return count,grid

    def dfs2(self, grid, r, c,i):
        grid[r][c] = -i
        directions = [(0,1), (0,-1), (-1,0), (1,0),(-1,-1),(1,1),(1,-1),(-1,1)]
        for d in directions:
            nr, nc = r + d[0], c + d[1]
            if self.is_valid(grid, nr, nc) and grid[nr][nc] == 1:
                self.dfs2(grid, nr, nc,i)

