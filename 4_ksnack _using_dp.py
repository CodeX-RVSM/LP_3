def knapsack_dp(values, weights, capacity):
    n = len(values)
    # Create a DP table initialized with 0
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build the table dp[][] in bottom-up manner
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    # Display result
    print("\nMaximum value in Knapsack =", dp[n][capacity])

    # Traceback to find which items are included
    w = capacity
    chosen_items = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen_items.append(i)
            w -= weights[i - 1]
    
    print("Items included (1-indexed):", chosen_items[::-1])
    return dp[n][capacity]


if __name__ == "__main__":
    n = int(input("Enter the number of items: "))

    values = []
    weights = []

    for i in range(n):
        v = int(input(f"Enter value of item {i+1}: "))
        w = int(input(f"Enter weight of item {i+1}: "))
        values.append(v)
        weights.append(w)

    capacity = int(input("Enter the maximum capacity of knapsack: "))

    knapsack_dp(values, weights, capacity)
