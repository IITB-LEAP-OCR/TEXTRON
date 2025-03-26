def calculate_average(file_path):
    """
    Calculate the average of float numbers from a file.
    Each number should be on a new line.
    """
    try:
        with open(file_path, 'r') as file:
            numbers = []
            for line in file:
                # Skip empty lines and try to convert each line to float
                if line.strip():
                    try:
                        number = float(line.strip())
                        numbers.append(number)
                    except ValueError:
                        print(f"Warning: Skipping invalid number: {line.strip()}")
                        continue
            
            if not numbers:
                print("No valid numbers found in the file.")
                return None
            
            average = sum(numbers) / len(numbers)
            return average
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    file_path = "log_cage_time_2.txt"  # Replace with your file path
    avg = calculate_average(file_path)
    
    if avg is not None:
        print(f"The average is: {avg:.2f}") 