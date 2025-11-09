def n_queens_with_first_fixed(n):
    col = set()
    posDiag = set()
    negDiag = set()
    res = []

    board = [["."] * n for _ in range(n)]

    # --- Place the first queen manually (for example at row 0, col 0) ---
    first_col = int(input(f"Enter the column (0 to {n-1}) for the first queen in row 0: "))
    board[0][first_col] = "Q"
    col.add(first_col)
    posDiag.add(0 + first_col)
    negDiag.add(0 - first_col)

    # Start from the next row (since row 0 is occupied)
    def backtrack(r):
        if r == n:
            res.append([" ".join(row) for row in board])
            return

        for c in range(n):
            if c in col or (r + c) in posDiag or (r - c) in negDiag:
                continue

            col.add(c)
            posDiag.add(r + c)
            negDiag.add(r - c)
            board[r][c] = "Q"

            backtrack(r + 1)

            col.remove(c)
            posDiag.remove(r + c)
            negDiag.remove(r - c)
            board[r][c] = "."

    backtrack(1)  # start from second row

    if not res:
        print(f"\nNo solutions exist for {n} queens with first queen at column {first_col}.")
    else:
        print(f"\nTotal solutions for {n} queens (first queen at column {first_col}): {len(res)}\n")
        for i, sol in enumerate(res, 1):
            print(f"Solution {i}:")
            for row in sol:
                print(row)
            print()

if __name__ == "__main__":
    n = int(input("Enter number of queens: "))
    if n <= 0:
        print("Please enter a positive integer.")
    else:
        n_queens_with_first_fixed(n)
