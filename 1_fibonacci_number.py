def recursive_fibonacci(n):
    if n <= 1:
        return n
    else:
        return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)

def non_recursive_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib

if __name__ == "__main__":
    n = int(input("Enter the number of terms: "))

    print("Recursive Fibonacci:")
    for i in range(n): 
        print(recursive_fibonacci(i), end=" ")
    print()

    print("Non-Recursive Fibonacci:")
    print(*non_recursive_fibonacci(n))
