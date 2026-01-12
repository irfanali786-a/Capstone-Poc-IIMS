def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def modulus(a, b):
    return a % b

def exponent(a, b):
    return a ** b

def floor_division(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a // b

if __name__ == "__main__":
    print("Mathematical Operations Program")
    a = int(input("Enter the first integer: "))
    b = int(input("Enter the second integer: "))

    print(f"Addition: {add(a, b)}")
    print(f"Subtraction: {subtract(a, b)}")
    print(f"Multiplication: {multiply(a, b)}")
    print(f"Division: {divide(a, b)}")
    print(f"Modulus: {modulus(a, b)}")
    print(f"Exponent: {exponent(a, b)}")
    print(f"Floor Division: {floor_division(a, b)}")
