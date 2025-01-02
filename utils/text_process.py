import os

def truncate_text(text, max_text_length):
    """
    Truncates text to the maximum allowed length based on words separated by " ".
    """
    tokens = text.split(" ") 
    if len(tokens) > max_text_length:
        tokens = tokens[:max_text_length]  
    return " ".join(tokens)  

def read_lines(filepath, max_text_length):
    """Reads each line from a file, ignoring empty lines and '#' comments."""
    if not os.path.isfile(filepath):
        return []
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove inline comment if present
            line = line.split('#', 1)[0].strip()
            if line:
                truncated_line = truncate_text(line, max_text_length)
                lines.append(truncated_line)
    return lines