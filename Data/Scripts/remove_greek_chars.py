import re
import string

def remove_non_ascii_and_greek(text):
    # 1. Normalize the text (e.g., replace accented chars with unaccented equivalents)
    #    This is helpful if you want to keep 'a' instead of 'á'.
    import unicodedata
    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # 2. Remove any remaining characters that are NOT:
    #    - Letters (a-z, A-Z)
    #    - Numbers (0-9)
    #    - Whitespace (\s)
    #    - Basic punctuation (defined by string.punctuation)
    
    # Create a string of all allowed characters
    allowed_chars = string.ascii_letters + string.digits + string.whitespace + string.punctuation
    
    # Filter the normalized text
    filtered_text = "".join(c for c in normalized_text if c in allowed_chars)
    
    # This step is often skipped if only basic cleaning is needed.
    return filtered_text

# --- Example Usage ---
input_file = 'wiki.txt'
output_file = 'cleaned_wiki.txt'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        cleaned_line = remove_non_ascii_and_greek(line)
        outfile.write(cleaned_line)

print(f"File cleaned and saved to {output_file}")